import flet as ft
import pandas as pd

from unimodal import UnimodalRetrievalSystem
from collections import defaultdict


DATA_ROOT = "./data"
id_information_df = pd.read_csv(f"{DATA_ROOT}/id_information_mmsr.tsv", sep="\t")

unimodal_rs = UnimodalRetrievalSystem(data_root=DATA_ROOT)

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
            ft.TextSpan("YAMEx", ft.TextStyle(size=30, weight=ft.FontWeight.BOLD)),
            ft.TextSpan("\nYet Another Music Explorer", ft.TextStyle(size=20)),
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

    def handle_search_now(e):
        song_query = search_song_field.value.split()
        artist_query = search_artist_field.value.strip()
        album_query = search_album_field.value.strip()
        current_algorithm = dropdown_algorithm.value
        current_number_results = current_slider_value

        results_song.controls.clear()

        query_id = "04OjszRi9rC5BlHC"
        #query_id = resolve_unimode_query_id(
        #    song=song_query,
        #    artist=artist_query,
        #    album_name=album_query
        #)
        if current_algorithm == "unimodal":
            ids, scores = unimodal_rs.retrieve(
                query_id =  query_id,
                modality = "audio",
                k_neighbors = current_number_results
            )
        print(f"Query: Artist: {artist_query}, Algorithm: {current_algorithm}, Number of results: {current_number_results}")
        page.update()



    def find_ids_by_artist(artist_query: str):
        if not artist_query:
            return []

        found_matches = id_information_df[
            id_information_df["artist"].str.contains(
                artist_query,  # string of search text
                case = False,  # False for case-insensitive, True for case-sensitive
                na = False,  #  ignore NaN values
                regex = False  # for . or *
            )
        ]
        return found_matches["id"].tolist()

    def find_ids_by_song(song_query: str):
        if not song_query:
            return []

        found_matches = id_information_df[
            id_information_df["song"].str.contains(
                song_query,  # string of search text
                case = False,  # False for case-insensitive, True for case-sensitive
                na = False,  #  ignore NaN values
                regex = False  # for . or *
            )
        ]
        return found_matches["id"].tolist()

    def find_ids_by_album(album_query: str):
        if not album_query:
            return []

        found_matches = id_information_df[
            id_information_df["album_name"].str.contains(
                album_query,  # string of search text
                case = False,  # False for case-insensitive, True for case-sensitive
                na = False,  #  ignore NaN values
                regex = False  # for . or *
            )
        ]
        return found_matches["id"].tolist()


    def find_ids(query:str, column:str):
        if not query:
            return []
        found_matches = id_information_df[
            id_information_df[column].str.contains(
                query,  # string of search text
                case = False,  # False for case-insensitive, True for case-sensitive
                na = False,  #  ignore NaN values
                regex = False  # for . or *
            )
        ]
        return found_matches["id"].tolist()

    def resolve_unimode_query_id(artist, song, album_name):
        if song:
            id = find_ids(song, "song")
            if id: return id
        if artist:
            id = find_ids(artist, "artist")
            if id: return id
        if album_name:
            id = find_ids(album_name, "album_name")
            if id: return id
        return []


    search_song_field = create_search_field(
        hint_text="Find song by title",
        on_submit_callback=handle_search_title
    )

    search_artist_field = create_search_field(
        hint_text="Find songs of an artist",
        on_submit_callback=handle_search_artist
    )

    search_album_field = create_search_field(
        hint_text="Find songs of an album",
        on_submit_callback=handle_search_album
    )

    create_search_title = ft.Column(
        col={"xs": 12, "sm": 6, "md": 4},
        controls=[
            ft.Text("Song:"),
            search_song_field
        ]
    )

    create_search_artist = ft.Column(
        col={"xs": 12, "sm": 6, "md": 4},
        controls=[
            ft.Text("Artist:"),
            search_artist_field
        ]
    )

    create_search_album = ft.Column(
        col={"xs": 12, "sm": 6, "md": 4},
        controls=[
            ft.Text("Album:"),
            search_album_field
        ]
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
        divisions=20,  # for step size of 5
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
            ft.dropdown.Option("random", "Random baseline"),
            ft.dropdown.Option("unimodal", "Unimodal"),
            ft.dropdown.Option("multi_early", "Multimodal - Early fusion"),
            ft.dropdown.Option("multi_late", "Multimodal - Late fusion"),
            ft.dropdown.Option("neural", "Neural-Network based")
        ],
        on_change=handle_dropdown_menu,
        col={"xs": 12, "sm": 6, "md": 4},
        border_radius=20,
        bgcolor=ft.Colors.DEEP_PURPLE_700,
        border_color=ft.Colors.DEEP_PURPLE_200,
        filled=True,
        fill_color=ft.Colors.DEEP_PURPLE_800,
        trailing_icon=ft.Icon(ft.Icons.ARROW_DROP_DOWN, color=ft.Colors.WHITE),
        expand=True
    )

    search_button = ft.ElevatedButton(
        text="Search Now",
        icon=ft.Icons.SEARCH,
        on_click=handle_search_now,
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.DEEP_PURPLE_800,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            side=ft.BorderSide(
                color=ft.Colors.DEEP_PURPLE_200,
                width=1
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=20),
        ),
        col={"xs": 6, "sm": 3, "md": 2},
    )


    control_row = ft.ResponsiveRow(
        controls=[
            slider_group,
            search_button,
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
