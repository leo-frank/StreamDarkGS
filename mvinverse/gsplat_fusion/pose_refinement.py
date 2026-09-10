"""Differentiable camera-space conversion for legacy 2DGS pose optimization."""
import torch
import torch.nn.functional as F


def quaternion_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, scalar-first (w, x, y, z)."""
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    av, bv = torch.broadcast_tensors(a[..., 1:], b[..., 1:])
    xyz = a[..., :1] * bv + b[..., :1] * av + torch.cross(av, bv, dim=-1)
    return torch.cat((w, xyz), dim=-1)


def rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert one 3x3 rotation matrix to a scalar-first quaternion.

    Camera base rotations are constants during optimization, so selecting the
    numerically stable branch with scalar comparisons does not cut a required
    gradient path.
    """
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {tuple(rotation.shape)}")
    r = rotation
    trace = float(torch.trace(r).detach())
    if trace > 0.0:
        s = torch.sqrt((torch.trace(r) + 1.0).clamp_min(1e-8)) * 2.0
        q = torch.stack(((s * 0.25), (r[2, 1] - r[1, 2]) / s,
                         (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s))
    else:
        diagonal = torch.diagonal(r)
        index = int(torch.argmax(diagonal).detach())
        if index == 0:
            s = torch.sqrt((1.0 + r[0, 0] - r[1, 1] - r[2, 2]).clamp_min(1e-8)) * 2.0
            q = torch.stack(((r[2, 1] - r[1, 2]) / s, s * 0.25,
                             (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s))
        elif index == 1:
            s = torch.sqrt((1.0 + r[1, 1] - r[0, 0] - r[2, 2]).clamp_min(1e-8)) * 2.0
            q = torch.stack(((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                             s * 0.25, (r[1, 2] + r[2, 1]) / s))
        else:
            s = torch.sqrt((1.0 + r[2, 2] - r[0, 0] - r[1, 1]).clamp_min(1e-8)) * 2.0
            q = torch.stack(((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                             (r[1, 2] + r[2, 1]) / s, s * 0.25))
    return F.normalize(q, dim=0, eps=1e-8)


def rotation_vector_to_matrix(rotation_delta: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues map with a stable small-angle expansion."""
    x, y, z = rotation_delta.unbind()
    zero = rotation_delta.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    theta_sq = rotation_delta.square().sum()
    theta = torch.sqrt(theta_sq.clamp_min(1e-16))
    small = theta_sq < 1e-8
    a = torch.where(small, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta)
    b = torch.where(
        small,
        0.5 - theta_sq / 24.0,
        (1.0 - torch.cos(theta)) / theta_sq.clamp_min(1e-16),
    )
    return torch.eye(3, device=rotation_delta.device, dtype=rotation_delta.dtype) + a * skew + b * (skew @ skew)


def incremental_world_to_camera(
    base_world_to_camera: torch.Tensor,
    rotation_delta: torch.Tensor,
    translation_delta: torch.Tensor,
) -> torch.Tensor:
    """Apply a camera-coordinate SE(3) increment to a base world-to-camera."""
    delta_rotation = rotation_vector_to_matrix(rotation_delta)
    result = torch.eye(4, device=base_world_to_camera.device, dtype=base_world_to_camera.dtype)
    result[:3, :3] = delta_rotation @ base_world_to_camera[:3, :3]
    result[:3, 3] = delta_rotation @ base_world_to_camera[:3, 3] + translation_delta
    return result


def camera_space_gaussians(means, quats, world_to_camera, rotation_delta, base_quaternion):
    """Transform fixed world Gaussians; gradients flow through means/quats, not viewmats.

    world_to_camera must equal Exp(delta) @ base_world_to_camera. base_quaternion
    is the scalar-first quaternion of base_world_to_camera's rotation.
    Scales and opacity remain unchanged under this rigid transform.
    """
    angle = rotation_delta.norm()
    dq = torch.cat((torch.cos(angle / 2).reshape(1),
                    rotation_delta * (0.5 * torch.sinc(angle / (2 * torch.pi)))))
    camera_quaternion = quaternion_product(dq, base_quaternion)
    return (
        means @ world_to_camera[:3, :3].T + world_to_camera[:3, 3],
        F.normalize(quaternion_product(camera_quaternion, quats), dim=-1, eps=1e-8),
    )
