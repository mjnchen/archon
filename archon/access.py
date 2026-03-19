"""Multi-tenancy and RBAC — tenant context and permission enforcement."""

from __future__ import annotations

import logging
from typing import Optional

from archon.exceptions import AccessDenied
from archon.types import Role, TenantContext

logger = logging.getLogger(__name__)


def require_permission(
    tenant: Optional[TenantContext],
    action: str,
) -> None:
    """Raise :class:`AccessDenied` if *tenant* lacks permission for *action*.

    If *tenant* is ``None``, the check is skipped (unscoped usage).
    """
    if tenant is None:
        return
    if not tenant.has_permission(action):
        raise AccessDenied(action, tenant.role.value)


def require_role(
    tenant: Optional[TenantContext],
    minimum_role: Role,
) -> None:
    """Raise if *tenant* role is below *minimum_role*.

    Role hierarchy: ADMIN > OPERATOR > VIEWER.
    """
    if tenant is None:
        return

    hierarchy = {Role.ADMIN: 3, Role.OPERATOR: 2, Role.VIEWER: 1}
    if hierarchy.get(tenant.role, 0) < hierarchy.get(minimum_role, 0):
        raise AccessDenied(
            f"minimum role {minimum_role.value}",
            tenant.role.value,
        )
