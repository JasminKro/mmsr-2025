import flet as ft
import pandas as pd

from unimodal import UnimodalRetrievalSystem
from enum import Enum

class RetrievalAlgorithms(str, Enum):
    RANDOM = "random"
    UNIMODAL = "unimodal"
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    NEUTRAL_NETWORK = "neutral_network"


DATA_ROOT = "./data"
id_information_df = pd.read_csv(f"{DATA_ROOT}/id_information_mmsr.tsv", sep="\t")

unimodal_rs = UnimodalRetrievalSystem(data_root=DATA_ROOT)
current_slider_value = 10 # default value
current_algorithm = RetrievalAlgorithms.RANDOM
retrieved_results = []

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

    slider_label = ft.Text(f"Number of results: {current_slider_value}", color=ft.Colors.WHITE)

    def handle_slider(e: ft.ControlEvent):
        global current_slider_value
        current_slider_value = int(e.control.value)
        slider_label.value = f" Number of results: {current_slider_value}"
        page.update()

    def handle_dropdown_menu(e):
        global current_algorithm
        print(repr(e.control.value))
        current_algorithm = dropdown_algorithm.value
        page.update()

    def create_search_field(
        hint_text: str = "Search for a song title, an artist or an album",
        expand=False,
        on_submit_callback=None
    ):
        return ft. TextField(
            hint_text=hint_text,
            bgcolor=ft.Colors.DEEP_PURPLE_50,
            border_radius=20,
            prefix_icon=ft.Icons.SEARCH_ROUNDED,
            expand=False,
            on_submit = lambda e: on_submit_callback(e.control.value)
                if on_submit_callback else None
        )

    def handle_search_now(e):
        query = search_field.value.strip()
        current_number_results = current_slider_value
        result_songs.controls.clear()
        retrieved_results.clear()
        query_id = resolve_unimode_query_id(query)

        if not query_id:
            result_songs.controls.append(
                ft.Text("Sorry, no results found", color=ft.Colors.WHITE)
            )
            page.update()
            return

        if current_algorithm == RetrievalAlgorithms.UNIMODAL:
            ids, scores = unimodal_rs.retrieve(
                query_id =  query_id,
                modality = "audio",
                k_neighbors = current_number_results
            )

        for i, (retrieved_id, score) in enumerate(zip(ids, scores)):
            res_row = id_information_df[id_information_df["id"] == retrieved_id]
            if res_row.empty: continue
            res = res_row.iloc[0]

            r = {
                "index": i + 1,  # running number
                "id": retrieved_id,
                "score": score,
                "song": res["song"],
                "artist": res["artist"],
                "album_name": res["album_name"]
            }
            retrieved_results.append(r)

            result_songs.controls.append(
                ft.ListTile(
                    hover_color=ft.Colors.DEEP_PURPLE_800,
                    title=ft.Text(
                        spans=[
                            ft.TextSpan(
                                f"[{r['index']}] ",
                                ft.TextStyle(color=ft.Colors.WHITE)
                            ),
                            ft.TextSpan(
                                r['song'],
                                ft.TextStyle(color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=18)
                            ),
                            ft.TextSpan(
                                f" - {r['artist']}, {r['album_name']} "
                                f"  |  score: {r['score']:.3f}",
                                ft.TextStyle(color=ft.Colors.WHITE)
                            )
                        ]
                    ),
                    on_click = lambda e, song=r: on_song_click(song)
            )
        )
        print (f"retrieved: {retrieved_results}")
        page.update()


    def find_id(query:str, column:str):
        if not query:
            return None
        found_matches = id_information_df[
            id_information_df[column].str.contains(
                query,  # string of search text
                case = False,  # False for case-insensitive, True for case-sensitive
                na = False,  #  ignore NaN values
                regex = False  # for . or *
            )
        ]
        if found_matches.empty:
            return None
        return found_matches.iloc[0]["id"]

    def resolve_unimode_query_id(query):
        search_order = ["song", "artist", "album_name"]

        for column in search_order:
            query_id = find_id(query, column)
            if query_id:
                return query_id
        return None

    def on_song_click(song_data):
        print("Clicked: ", song_data)


    search_field = create_search_field(
    )

    results_slider = ft.Slider(
        min=1,
        max=100,
        divisions=20,  # for step size of 5
        value=current_slider_value,
#        label="{value}",
        on_change=handle_slider,
        expand=True
    )

    slider_group = ft.Row(
        controls=[
            slider_label,
            results_slider
        ]
    )

    dropdown_algorithm = ft.Dropdown(
        label="Algorithm",
        label_style=ft.TextStyle(color=ft.Colors.WHITE),
        text_style=ft.TextStyle(color=ft.Colors.WHITE),
        value=RetrievalAlgorithms.RANDOM,  # start value
        options=[
            ft.dropdown.Option(RetrievalAlgorithms.RANDOM.value, "Random baseline"),
            ft.dropdown.Option(RetrievalAlgorithms.UNIMODAL.value, "Unimodal"),
            ft.dropdown.Option(RetrievalAlgorithms.EARLY_FUSION.value, "Multimodal - Early fusion"),
            ft.dropdown.Option(RetrievalAlgorithms.LATE_FUSION.value, "Multimodal - Late fusion"),
            ft.dropdown.Option(RetrievalAlgorithms.NEUTRAL_NETWORK.value, "Neural-Network based")
        ],
        on_change=handle_dropdown_menu,
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
        col={"xs": 8, "sm": 4, "md": 3},
    )


    control_row = ft.ResponsiveRow(
        controls=[
            ft.Container(
                content=search_field,
                alignment=ft.alignment.center_right,
                col={"xs": 12, "md": 6},
            ),
            ft.Container(
                content=dropdown_algorithm,
                alignment=ft.alignment.center_left,
                col={"xs": 12, "md": 2},
            ),
            ft.Container(
                content=slider_group,
                alignment=ft.alignment.center_left,
                col={"xs": 12, "md": 4},
            )
        ],
        width=float("inf"),
        spacing=20
    )

    result_songs = ft.Column(scroll="auto", expand=True)

    intermediate_results_container = ft.Container(
        content=result_songs,
        padding=15,
        border=ft.border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        expand=True,
        alignment=ft.alignment.top_left,
        col={"xs": 12, "md": 6}  # xs = small monitor: full width, md = medium = 6 of 12 colums width
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
                control_row,
                search_button,
                ft.Text("Top results:"),
                result_row
            ],
            horizontal_alignment="center",
            spacing=25,
            expand=True
        )
    )

ft.app(target=main, view=ft.WEB_BROWSER)
