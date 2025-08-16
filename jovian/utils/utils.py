import torch
from einops import einsum, rearrange

from jaxtyping import Float, Array, Int
import argparse
from dataclasses import fields

import numpy as np
from numpy import ndarray
from jovian.constants.constants import one_to_three, aa_to_bb_coord, BBHeavyAtom, AA

from pathlib import Path
import subprocess
import os
import shutil

from biotite.structure.io.pdb import PDBFile
from biotite.structure import Atom, AtomArray, array

import yaml
from jovian.config.train import TrainConfig

def load_cfg(path: str | None) -> TrainConfig:
    cfg = TrainConfig()
    if path is None:
        return cfg
    with open(path) as f:
        overrides = yaml.safe_load(f)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown field in cfg: {k}")
        setattr(cfg, k, v)
    return cfg


def get_frame_between_pts(v_local: Float[Array, 'l a 3'], v_global: Float[Array, 'l a 3']):
    """ Uses kabsch to get the frame relating two points, i.e., 

    T v_local = v_global

    v_local: length(*), atom(1:3), pos(1:3)
    """
    
    centroid_local = v_local.mean(axis=-2, keepdim=True)
    centroid_global = v_global.mean(axis=-2, keepdim=True)

    v_local_centered = v_local - centroid_local
    v_global_centered = v_global - centroid_global
    

    H = v_global_centered.transpose(-1,-2) @ v_local_centered
    U, S, Vt = torch.linalg.svd(H) 
    d = torch.det(torch.matmul(Vt.transpose(-1, -2), U.transpose(-1, -2)))  # B
    Vt_new = Vt.clone()
    flip = d < 0.0

    if flip.any().item():
        Vt_new[flip, -1] = Vt[flip, -1] * -1.0
    R = U @ Vt_new

    t = centroid_global.squeeze(-2) - (R @ centroid_local.transpose(-1,-2)).squeeze(-1)
    return R, t

def get_true_frame(pos: Float[Array, 'l 4 3'], tokens: Int[Array, 'l']):
    """ Returns the true frame.  

    Though in principle one can compute true frames directly from the dihedral angles, 
    this ends up being slightly inaccurate and causes problems at the edge. This function
    will return the ground truth frames for [N, CA, C, O] provided as positions.

    This only works in torch, but I precompute these anyway....
    """
    assert len(tokens.shape) == 1 # no batch!
    x_local = aa_to_bb_coord[tokens]
    T_bb = make_frame(*get_frame_between_pts(x_local[:, :-1, :], pos[:, :-1, :]))

    T_psi = get_psi_frame()[tokens]

    # now map from o_local to o_global
    o_local = x_local[:, -1, :]
    o_global = pos[:, -1, :]

    o_global_prime = apply_frame((T_bb @ T_psi).inverse(), o_global)
    y1, z1 = o_local[..., 1], o_local[..., 2]
    y2, z2 = o_global_prime[..., 1], o_global_prime[..., 2]

    numer = y1 * z2 - z1 * y2
    denom = y1 * y2 + z1 * z2
    theta = torch.atan2(numer, denom).unsqueeze(-1)
    alpha = torch.cat((theta.sin(), theta.cos()), dim=-1)
    T_torsion = make_frame(make_rotation_around_x(alpha), None)

    T_total = T_bb.unsqueeze(-3).repeat(1, 4, 1, 1)
    T_total[:, -1] = T_total[:, -1] @ T_psi @ T_torsion

    return T_total, T_bb, T_psi, T_torsion, alpha
    

def get_backbone_frame(pos: Float[Array, '... 4 3']):
    # This ordering is important! Must match literature positions in aa_to_bb_coord, i.e., 
    # T_bb @ x_lit = pos
    R_bb = gram_schmidt(pos[..., BBHeavyAtom.C, :]-pos[..., BBHeavyAtom.CA, :],
            pos[..., BBHeavyAtom.N, :]-pos[..., BBHeavyAtom.CA, :])

    T_bb = make_frame(R_bb, pos[..., BBHeavyAtom.CA, :])
    return T_bb

def get_psi_frame():
    R_psi = gram_schmidt(
        aa_to_bb_coord[:, BBHeavyAtom.C] - aa_to_bb_coord[:, BBHeavyAtom.CA],
        aa_to_bb_coord[:, BBHeavyAtom.CA] - aa_to_bb_coord[:, BBHeavyAtom.N],
    )
    t_psi = aa_to_bb_coord[:, BBHeavyAtom.C]

    # There is not a ton of variation, so we'll set UNK to ALA
    # add PAD to something reasonable because it shouldn't matter if we mask correctly,
    # but we need to be able to take inverses

    R_psi[AA.UNK] = R_psi[AA.ALA].clone()
    R_psi[AA.PAD] = R_psi[AA.ALA].clone()
    t_psi[AA.UNK] = t_psi[AA.ALA].clone()
    t_psi[AA.PAD] = t_psi[AA.ALA].clone()
    return make_frame(R_psi, t_psi)



# frame stuff
def get_empty_frames(sizes, device, dtype):
    T = torch.zeros((*sizes, 4, 4), dtype=dtype, device=device)
    T[..., 0, 0] = 1
    T[..., 1, 1] = 1
    T[..., 2, 2] = 1
    T[..., 3, 3] = 1
    return T

def make_frame(R, t):
    sizes = R.shape[:-2]
    if t is None:
        t = torch.zeros(sizes + (3,), dtype=R.dtype, device=R.device)
    T = torch.zeros((*sizes, 4, 4), dtype=R.dtype, device=R.device)
    T[..., :-1, :-1] = R
    T[..., :-1, -1] = t
    T[..., -1, -1] = 1
    return T

def make_rotation_around_x(alpha):
    sin, cos = alpha.unbind(-1)
    Rtorsion = torch.zeros((*alpha.shape[:-1], 3, 3), device=alpha.device, dtype=alpha.dtype)
    Rtorsion[..., 0, 0] = 1
    Rtorsion[..., 1, 1] = cos
    Rtorsion[..., 2, 2] = cos
    Rtorsion[..., 1, 2] = -sin
    Rtorsion[..., 2, 1] = sin
    return Rtorsion

def frame_from_vec6(vec6):
    bi, ci, di, ti = vec6.split([1,1,1,3], dim=-1)
    v = torch.cat(
        [torch.ones_like(bi, device=bi.device), bi, ci, di], dim=-1
    ) # b l 4
    v = v / v.norm(dim=-1, keepdim=True)
    ai, bi, ci, di = v.unbind(dim=-1)
    Ri = torch.zeros(ai.shape + (3,3), dtype=ai.dtype, device=ai.device)
    Ri[..., 0, 0] = ai**2 + bi**2 - ci**2 - di**2
    Ri[..., 0, 1] = 2 * (bi*ci - ai*di)
    Ri[..., 0, 2] = 2 * (bi*di + ai*ci)
    Ri[..., 1, 0] = 2 * (bi*ci + ai*di)
    Ri[..., 1, 1] = ai*ai - bi*bi + ci*ci - di*di
    Ri[..., 1, 2] = 2 * (ci*di - ai*bi)
    Ri[..., 2, 0] = 2 * (bi*di - ai*ci)
    Ri[..., 2, 1] = 2 * (ci*di + ai*bi)
    Ri[..., 2, 2] = ai*ai - bi*bi - ci*ci + di*di
    return make_frame(Ri, ti)

def rbf(D: Float[Array, "b l k"], D_min=2.0, D_max=22.0, D_count=16):
    if D.ndim == 3:
        D = D.unsqueeze(-1)
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)  # C
    D_mu = rearrange(D_mu, "c -> () () () c")

    D_sigma = (D_max - D_min) / D_count
    # D = D.unsqueeze(-1)
    # b l k 1 - 1 1 1 c
    D_cent = rearrange(
        D.unsqueeze(-1) - D_mu.unsqueeze(-2), 
        'b l k a d -> b l k (a d)'
    ) # (batch) (index) (neighbor) (atom-atom * dim)

    rbf = (-((D_cent / D_sigma) ** 2)).exp()  # b l k c
    return rbf 


def apply_frame(T, coords, pdims=[]):
    """ applies frame to coords. pdims is all dimensions to parallelize over """
    squeeze = not pdims
    if squeeze:
        coords = coords.unsqueeze(-2)
        pdims = [-2]

    assert coords.size(-1) == 3
    shape = coords.shape
    pdims = [len(coords.shape) + p if p < 0 else p for p in pdims]
    mdims = [p for p in range(len(shape)) if p not in pdims]

    mshape = [shape[s] for s in mdims]
    pshape = [shape[s] for s in pdims]

    # can't parallelize over spatial dimension
    assert len(shape) - 1 not in pdims

    new_order = mdims + pdims
    coords = coords.permute(new_order).contiguous().view(*[shape[s] for s in mdims], -1)
    coords = torch.cat((coords, torch.ones_like(coords)[..., [0], :]), dim=-2)

    out = (T @ coords)[..., :-1, :] # keep only spatial part
    out = out.view(*(mshape[:-1] + [3] + pshape))
    out = out.permute(torch.argsort(torch.tensor(new_order)).tolist())

    if squeeze:
        out = out.squeeze(-2)
    return out


def inverse(T):
    R = T[..., :-1, :-1].transpose(-1, -2)
    t = T[..., :-1, -1]
    t = -(R @ t.unsqueeze(-1)).squeeze(-1)
    return make_frame(R, t)


def make_frame_np(R, t):
    sizes = R.shape[:-2]
    if t is None:
        t = np.zeros(sizes + (3,), dtype=R.dtype)
    T = np.zeros((*sizes, 4, 4), dtype=R.dtype)
    T[..., :-1, :-1] = R
    T[..., :-1, -1] = t
    T[..., -1, -1] = 1
    return T

def apply_frame_np(T: Float[Array, '... 4 4'], coords: Float[Array, '... 3']):
    sizes = coords.shape[:-1]
    coords = np.cconcatenate((coords, np.ones(sizes + (1,), dtype=coords.dtype)), dim=-1)
    return (T @ coords.transpose(-1,-2)).transpose(-1,-2)[..., :-1]

def get_dihedrals(
    a: Float[Array, '... 3'], 
    b: Float[Array, '... 3'], 
    c: Float[Array, '... 3'], 
    d: Float[Array, '... 3']
):

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= (np.linalg.norm(b1, axis=-1)[:,None] + 1.e-8)

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

########################## end frame stuff #######################

def cluster_mmseqs(pth_to_fasta, sequence_sim: float = 0.5, silent=True):
    """
    """
    pth_to_fasta = Path(pth_to_fasta).resolve()
    parent = str(pth_to_fasta.parent)

    cmd = (
        f"docker run --rm -v {parent}:/data -u $(id -u):$(id -g) "
        f"ghcr.io/soedinglab/mmseqs2:latest easy-cluster /data/{pth_to_fasta.name} "
        f"/data/clusterres /data/scrap --min-seq-id {sequence_sim} -c 0.8 --cov-mode 1"
    )

    stdout = subprocess.DEVNULL if silent else None
    stderr = subprocess.DEVNULL if silent else None

    # Run the command
    subprocess.run(cmd, shell=True, stdout=stdout, stderr=stderr)

    shutil.rmtree(f'{parent}/scrap')
    clusters = np.loadtxt(f"{parent}/clusterres_cluster.tsv", delimiter="\t", dtype=str)
    os.remove(f"{parent}/clusterres_all_seqs.fasta")
    os.remove(f"{parent}/clusterres_cluster.tsv")
    os.remove(f"{parent}/clusterres_rep_seq.fasta")


    return clusters

# align function
def align_structures(P: Float[Array, 'l c 3'], Q: Float[Array, 'l c 3']):
    if isinstance(P, np.ndarray):
        P = torch.from_numpy(P).to(torch.float32)
    if isinstance(Q, np.ndarray):
        Q = torch.from_numpy(Q).to(torch.float32)
    P, Q = P.unsqueeze(0), Q.unsqueeze(0)
    _, R, _ = compute_rmsd_with_kabsch(P, Q)
    P -= P.mean(dim=1, keepdim=True)
    Q -= Q.mean(dim=1, keepdim=True)
    P = einsum(R, P, 'x y, b l c y -> b l c x').squeeze(0)
    Q = Q.squeeze(0)
    return P.numpy(), Q.numpy()

# lifted from tokenizers

@torch.no_grad()
def compute_rmsd_with_kabsch(
    P: Float[Array, "b l c 3"], Q: Float[Array, "b l c 3"]):
    if isinstance(P, np.ndarray):
        P = torch.from_numpy(P).to(torch.float32)
    if isinstance(Q, np.ndarray):
        Q = torch.from_numpy(Q).to(torch.float32)
    P = P.to(Q.device)

    P = P.reshape(1, -1, 3)
    Q = Q.reshape(1, -1, 3)
    mask = torch.ones(P.shape[:-1], device=P.device).bool()
    assert P.shape == Q.shape, "Matrix dimensions must match"
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(-1)

    # zero out mask
    P = torch.where(mask, P, 0.0)
    Q = torch.where(mask, Q, 0.0)

    # Compute centroids
    centroid_P = P.sum(dim=-2, keepdim=True) / mask.sum(dim=1, keepdim=True)
    centroid_Q = Q.sum(dim=-2, keepdim=True) / mask.sum(dim=1, keepdim=True)

    t = (centroid_Q - centroid_P).squeeze(1)

    # center clouds
    p = torch.where(mask, P - centroid_P, 0.0)  # BxNx3
    q = torch.where(mask, Q - centroid_Q, 0.0)  # BxNx3

    # Compute covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0

    Vt_new = Vt.clone()
    if flip.any().item():
        Vt_new[flip, -1] = Vt[flip, -1] * -1.0

    # Optimal rotation
    R = torch.matmul(Vt_new.transpose(1, 2), U.transpose(1, 2))

    z = (torch.where(mask, einsum(p, R, "b l x, b y x -> b l y") - q, 0) ** 2).sum(
        dim=(1, 2), keepdim=True
    )
    rmsd = (z / mask.sum(dim=1, keepdim=True)).sqrt().squeeze(1, 2)

    # note: this is bad but I really need to debug some issues around here and it's convenient

    return rmsd.mean(), R.squeeze(0), t.squeeze(0)

def add_oxygen_atom(coords: Float[Array, '... 3 3'], coords_with_o: Float[Array, '... 4 3'], tokens=None):
    """
    The ESM tokenizer does not manifestly predict oxygen atoms (the paper claims it does, but
    I haven't been able to get it to work, though this could be a wrong convention, and the angle
    prediction head isn't used on the repo, so I can't be sure it's even properly trained).

    As an interim fix, because visualization and proteinMPNN both expect oxygen, we can take the 
    relative oxygen frame from the predicted coordinate, then apply it to the decoded backbone frame 
    to get out oxygen atoms.
    """
    if isinstance(coords_with_o, np.ndarray):
        coords_with_o = torch.from_numpy(coords_with_o).to(torch.float32)
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).to(torch.float32)
    


    coords_with_o = coords_with_o.clone()

    if tokens is None:
        tokens = torch.full(coords_with_o.shape[:-2], fill_value=AA.ALA)
    if isinstance(tokens[0], str):
        tokens = torch.tensor([AA[one_to_three[t]] for t in tokens], device=coords.device, dtype=torch.long)

    T_total, _, T_psi, T_torsion, alpha = get_true_frame(coords_with_o, tokens)
    T_bb = get_backbone_frame(coords)
    T_o = T_bb @ T_psi @ T_torsion
    pos_o = apply_frame(T_o, aa_to_bb_coord[tokens][..., BBHeavyAtom.O, :])
    coords_with_o = torch.cat((coords, pos_o.unsqueeze(-2)), dim=-2)
    return coords_with_o



def write_pdb(pos, seq, output_file, atoms_to_save=['N', 'CA', 'C']):
    atoms = []
    f = PDBFile()
    for l in range(pos.shape[0]):
        for m in range(len(atoms_to_save)):
            res_name = one_to_three[seq[l]]
            atoms.append(Atom(pos[l, m], atom_name=BBHeavyAtom(m).name, chain_id='A', res_name=res_name, element=BBHeavyAtom(m).name[0], res_id=l))
    f.set_structure(array(atoms))
    f.write(output_file)

def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1) :]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)



def gram_schmidt(x, y):
    """ x = x axis, """
    x = x / (x.norm(dim=-1, keepdim=True) + 1.e-8)
    e1 = y - einsum(x, y, '... d, ... d -> ...').unsqueeze(-1) * x
    e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1.e-8)
    e2 = torch.linalg.cross(x, e1)
    x, e1, e2 = map(lambda arr: arr.unsqueeze(-2), (x, e1, e2))
    R = torch.cat([x, e1, e2], dim=-2).transpose(-1,-2).contiguous()
    return R

# special frame functions . should eventually be merged with extensive testing
def query_local_to_global(
    x: Float[Array, "b l q 3"], R: Float[Array, "b l 3 3"], t: Float[Array, "b l 3"]
):
    return einsum(R, x, "b l x y, b l q y -> b l q x") + rearrange(
        t, "b l x -> b l () x"
    )

def local_to_global(x: Float[Array, '... l 3'], R: Float[Array, '... l 3 3'], t: Float[Array, '... l 3']):
    return einsum(R, x, '... x y, ... y -> ... x') + t

def global_to_local(x, R, t):
    return einsum(R.transpose(-1, -2), x - t, "... x y, ... y -> ... x")


def pairwise_global_to_local(
    x: Float[Array, "... n 3"], R: Float[Array, "... f 3 3"], t: Float[Array, "... f 3"]
) -> Float[Array, "... n f 3"]:
    diff = x.unsqueeze(-2) - t.unsqueeze(-3)  # Npts, Nframes
    return einsum(R.transpose(-1, -2), diff, "... f x y, ... p f y -> ... p f x")


def quaternion_to_rotation_matrix(q):
    """
    Convert a normalized quaternion to a 3x3 rotation matrix with support for broadcasting.

    Parameters:
    q : Tensor
        Normalized quaternion tensor with shape (..., 4),
        where the last dimension represents [q0, q1, q2, q3].

    Returns:
    R : Tensor
        Rotation matrix tensor with shape (..., 3, 3).
    """
    q = q / torch.linalg.norm(
        q, dim=-1, keepdim=True
    )  # Ensure quaternion is normalized
    q0, q1, q2, q3 = q.unbind(-1)

    # Compute rotation matrix elements
    R = torch.empty((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (q2**2 + q3**2)
    R[..., 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[..., 0, 2] = 2 * (q1 * q3 + q0 * q2)
    R[..., 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[..., 1, 1] = 1 - 2 * (q1**2 + q3**2)
    R[..., 1, 2] = 2 * (q2 * q3 - q0 * q1)
    R[..., 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[..., 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[..., 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return R


