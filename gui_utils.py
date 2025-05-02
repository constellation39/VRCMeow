import logging

import flet as ft

logger = logging.getLogger(__name__)


# --- Banner Helpers ---
def _close_banner_internal(page_ref: ft.Page, e: ft.ControlEvent = None):
    """Internal function to close the banner and update the page."""
    if page_ref and page_ref.banner:
        page_ref.banner.open = False
        try:
            if page_ref.controls:
                page_ref.update()
            else:
                logger.warning(
                    "Page has no controls, skipping update in _close_banner_internal."
                )
        except Exception as ex:
            logger.error(
                f"Error updating page while closing banner: {ex}", exc_info=True
            )
    elif page_ref:
        logger.debug("Close banner called but no banner exists.")
    # else: page_ref is None, cannot close


def close_banner(page_ref: ft.Page):
    """Closes the currently open banner on the page. (Public wrapper)"""
    _close_banner_internal(page_ref)


def show_banner(
    page_ref: ft.Page,
    message: str,
    icon: str = ft.icons.INFO_OUTLINE,
    icon_color: str = ft.colors.BLUE_800,
    bgcolor: str = ft.colors.BLUE_100,
    text_color: str = ft.colors.BLACK,
    button_color: str = ft.colors.BLUE_900,
):
    """Displays a banner with a message and a close button."""
    if not page_ref:
        logger.error(f"Cannot show banner, page_ref is None. Message: {message}")
        return

    page_ref.banner = ft.Banner(
        bgcolor=bgcolor,
        leading=ft.Icon(icon, color=icon_color),
        content=ft.Text(message, color=text_color),
        actions=[
            ft.TextButton(
                "关闭",
                on_click=lambda e: _close_banner_internal(
                    page_ref, e
                ),  # Use internal closer
                style=ft.ButtonStyle(color=button_color),
            )
        ],
    )
    page_ref.banner.open = True
    try:
        if page_ref.controls:
            page_ref.update()
        else:
            logger.warning(
                "Page has no controls, skipping update after showing banner."
            )
    except Exception as e:
        logger.error(f"Error updating page after showing banner: {e}", exc_info=True)


def show_error_banner(page_ref: ft.Page, message: str):
    """Displays an error banner."""
    show_banner(
        page_ref,
        message,
        icon=ft.icons.ERROR_OUTLINE,
        icon_color=ft.colors.RED_800,
        bgcolor=ft.colors.RED_100,
        button_color=ft.colors.RED_900,
    )


def show_success_banner(page_ref: ft.Page, message: str):
    """Displays a success banner."""
    show_banner(
        page_ref,
        message,
        icon=ft.icons.CHECK_CIRCLE_OUTLINE,
        icon_color=ft.colors.GREEN_800,
        bgcolor=ft.colors.GREEN_100,
        button_color=ft.colors.GREEN_900,
    )


# Add other potential utility functions here if needed (e.g., confirmation dialogs)
