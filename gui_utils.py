import flet as ft

# --- Helper function to close banner ---
def close_banner(page_ref: ft.Page):
    """Closes the currently open banner on the page."""
    if page_ref.banner:
        page_ref.banner.open = False
        page_ref.update()

# Add other potential utility functions here if needed
