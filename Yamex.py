import flet as ft

def main(page: ft.Page):
    page.padding = 20
    page.bgcolor = ft.Colors.DEEP_PURPLE_900
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.DEEP_PURPLE_300,
            secondary=ft.Colors.DEEP_PURPLE_200,
            background=ft.Colors.DEEP_PURPLE_900,
            surface=ft.Colors.DEEP_PURPLE_800,
            on_primary=ft.Colors.WHITE,
            on_secondary=ft.Colors.WHITE,
            on_background=ft.Colors.WHITE,
            on_surface=ft.Colors.WHITE
        ),
        text_theme=ft.TextTheme(
            title_large=ft.TextStyle(
                size=30,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.WHITE
            ),
            body_medium=ft.TextStyle(
                size=16,
                color=ft.Colors.WHITE
            ),
        ),
        slider_theme=ft.SliderTheme(
            active_track_color=ft.Colors.DEEP_PURPLE_200,
            inactive_track_color=ft.Colors.DEEP_PURPLE_50,
            thumb_color=ft.Colors.DEEP_PURPLE_300,
        )
    )


    # title
    title = ft.Text(
        spans=[
            ft.TextSpan("Music ABC ", ft.TextStyle(size=30, weight=ft.FontWeight.BOLD)),
            ft.TextSpan("\nabcdfghijk", ft.TextStyle(size=20)),
        ],
        text_align=ft.TextAlign.CENTER
    )
    title_placement = ft.Row(
        [title],
        alignment=ft.MainAxisAlignment.CENTER,  # horizontal alignment of children
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True
    )

    # functions for search fields
    def handle_search_title(query: str):
        # TODO call song title search (print is only placeholder)
        print("search by title in progress")

    def handle_search_artist(query: str):
        # TODO call song artist search (print is only placeholder)
        print("search by artist progress")

    def handle_search_album(query: str):
        # TODO call song album search (print is only placeholder)
        print("search by album progress")

    slider_label = ft.Text("Number of results:", color=ft.Colors.WHITE)

    def handle_slider(e: ft.ControlEvent):
        slider_label.value = f" Number of results: {int(e.control.value)}"
        page.update()

    current_slider_value = 10  # default value
    slider_label = ft.Text(f"Number of results: {current_slider_value}", color=ft.Colors.WHITE)

    def handle_dropdown_menu(e):
        global current_slider_value
        current_slider_value = int(e.control.value)
        slider_label.value = f"Number of results: {current_slider_value}"
        page.update()

    def create_search_field(
        hint_text: str = "",
        on_submit_callback=None
    ):
        return ft. TextField(
            hint_text=hint_text,
            bgcolor=ft.Colors.DEEP_PURPLE_50,
            border_radius=20,
            prefix_icon=ft.Icons.SEARCH_ROUNDED,
            col = {"xs": 12, "sm": 6, "md": 4},
            on_submit = lambda e: on_submit_callback(e.control.value)
                if on_submit_callback else None
        )

    create_search_title = create_search_field(
        hint_text="Find song by title",
        on_submit_callback=handle_search_title
    )

    create_search_artist = create_search_field(
        hint_text="Find songs of an artist",
        on_submit_callback=handle_search_artist
    )
    create_search_album = create_search_field(
        hint_text="Find songs of an album",
        on_submit_callback=handle_search_album
    )

    search_fields = ft.ResponsiveRow(
        controls=[
            create_search_title,
            create_search_artist,
            create_search_album,
        ],
    )

    results_slider = ft.Slider(
        min=1,
        max=100,
        divisions=99,  # for step size of 1
        value=current_slider_value,
        label="",
        on_change=handle_slider,
        col={"xs": 12, "sm": 6, "md": 4},
    )

    slider_group = ft.Column(
        controls=[
            slider_label,
            results_slider
        ],
        col={"xs": 12, "sm": 6, "md": 4}
    )

    dropdown_algorithm = ft.Dropdown(
        label="Algorithm",
        label_style=ft.TextStyle(color=ft.Colors.WHITE),
        text_style=ft.TextStyle(color=ft.Colors.WHITE),
        hint_text="choose an Algorithm",
        value="A",  # start value
        options=[
            ft.dropdown.Option("A", "Random baseline"),
            ft.dropdown.Option("B", "Unimodal"),
            ft.dropdown.Option("C", "Multimodal - Early fusion"),
            ft.dropdown.Option("D", "Multimodal - Late fusion"),
            ft.dropdown.Option("E", "Neural-Network based")
        ],
        on_change=handle_dropdown_menu,
        col={"xs": 12, "sm": 6, "md": 4},
        border_radius=20,
        bgcolor=ft.Colors.DEEP_PURPLE_700,
        border_color=ft.Colors.DEEP_PURPLE_200,
        expand=True
    )

    control_row = ft.ResponsiveRow(
        controls=[
            slider_group,
            dropdown_algorithm,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        spacing=20
    )

    dummy_songs = [
        "Dummy song A - Artist 1, Album Z, 2025",
        "Dummy song B - Artist 2, Album Y, 1983",
        "Dummy song C - Artist 3, Album X, 2023",
        "Dummy song D - Artist 4, Album W, 1993",
        "Dummy song E - Artist 3, Album V, 2021",
        "Dummy song F - Artist 1, Album U, 2003",
        "Dummy song G - Artist 5, Album T, 2010",
        "Dummy song H - Artist 6, Album S 1999",
        "Dummy song I - Artist 7, Album R, 2024",
        "Dummy song J - Artist 8, Album Q, 1978",
    ]

    # dummy results
    results_song = ft.Column(scroll="auto", expand=True)

    # initial songs for demo
    results_song.controls.append(ft.Text("initial results for demo:", color=ft.Colors.WHITE))
    for song in dummy_songs:
         results_song.controls.append(ft.Text(song, color=ft.Colors.WHITE))

    intermediate_results_container = ft.Container(
        content=results_song,
        padding=15,
        border=ft.border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        expand=True,
        alignment=ft.alignment.top_left,
        col={"xs": 12, "md": 6}  # xs = small monitor: voll width, md = medium = 6 of 12 colums width
    )

    result_container = ft.Container(
        content=ft.Text("Details", color=ft.Colors.WHITE),
        padding=15,
        border=ft.border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        expand=True,
        alignment=ft.alignment.top_left,
        col={"xs": 12, "md": 6}
    )

    result_row = ft.ResponsiveRow(
        controls=[
            intermediate_results_container,
            result_container
        ],
        spacing=25,
        alignment=ft.MainAxisAlignment.START
    )

    page.add(
        ft.Column(
            controls=[
                title,
                search_fields,
                control_row,
                ft.Text("Top results:"),
                result_row
            ],
            horizontal_alignment="center",
            spacing=25,
            expand=True
        )
    )

ft.app(target=main, view=ft.WEB_BROWSER)
