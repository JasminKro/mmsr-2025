import flet as ft
import pandas as pd

from enum import Enum
from strategies import RandomStrategy, UnimodalStrategy
from unimodal import UnimodalRetrievalSystem, Evaluator

class RetrievalAlgorithms(str, Enum):
    RANDOM = "random"
    UNIMODAL = "unimodal"
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    NEUTRAL_NETWORK = "neutral_network"


DATA_ROOT = "./data"
id_information_df = pd.read_csv(f"{DATA_ROOT}/id_information_mmsr.tsv", sep="\t")
id_url_df = pd.read_csv(f"{DATA_ROOT}/id_url_mmsr.tsv", sep="\t")


# Lookup dictionary
# This maps 'id' -> {'song': ..., 'artist': ..., 'album_name': ..., 'url': ...}
song_lookup_dict = id_information_df.merge(id_url_df, on="id", how="left").set_index("id").to_dict("index")

evaluator = Evaluator(DATA_ROOT)
unimodal_rs = UnimodalRetrievalSystem(DATA_ROOT, evaluator)
current_slider_value = 10 # default value
current_algorithm = RetrievalAlgorithms.RANDOM
retrieved_results = []

def main(page: ft.Page):
    page.scroll = ft.ScrollMode.AUTO
    page.padding = 20
    page.bgcolor = ft.Colors.DEEP_PURPLE_900
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.DEEP_PURPLE_300,
            secondary=ft.Colors.DEEP_PURPLE_200,
            surface_container=ft.Colors.DEEP_PURPLE_900,
            surface=ft.Colors.DEEP_PURPLE_800,
            on_primary=ft.Colors.WHITE,
            on_secondary=ft.Colors.WHITE,
            on_surface_variant=ft.Colors.WHITE,
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

    def handle_slider(e: ft.ControlEvent):#
        global current_slider_value
#        if isinstance(e.control, ft.Slider):
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
        on_submit_callback=None
    ):
        return ft. TextField(
            hint_text=hint_text,
            hint_style=ft.TextStyle(color=ft.Colors.DEEP_PURPLE_800),
            bgcolor=ft.Colors.DEEP_PURPLE_50,
            border_radius=20,
            color=ft.Colors.BLACK,
            prefix_icon=ft.Icon(ft.Icons.SEARCH_ROUNDED, color=ft.Colors.DEEP_PURPLE_800),
            expand=True,
            on_submit = lambda e: on_submit_callback(e.control.value)
                if on_submit_callback else None
        )

    def handle_search_now(e):
        query = search_field.value.strip()
        query_id = resolve_unimode_query_id(query)

        result_songs.controls.clear()

        if not query_id:
            result_songs.controls.append(ft.Text("No results found for that query.", color="red"))
            page.update()
            return

        # Select the strategy based on the dropdown value
        selected_algo_key = RetrievalAlgorithms(dropdown_algorithm.value)
        strategy = strategies.get(selected_algo_key)

        if not strategy:
            result_songs.controls.append(ft.Text("Algorithm not yet implemented.", color="yellow"))
            page.update()
            return

        # execute search
        ids, scores = strategy.search(query_id, current_slider_value)

        # update UI
        retrieved_results.clear()
        for i, (ret_id, score) in enumerate(zip(ids, scores)):
            res = song_lookup_dict.get(ret_id)

            if not res:
                continue

            song_info = {
                "index": i + 1,
                "id": ret_id,
                "score": score,
                "song": res["song"],
                "artist": res["artist"],
                "album_name": res["album_name"],
                "url": res["url"]
            }
            retrieved_results.append(song_info)

            result_songs.controls.append(
                ft.ListTile(
                    leading=ft.Text(f"[{song_info['index']}]", color="white70", size=14),
                    title=ft.Text(f"{song_info['song']}", color="white", weight=ft.FontWeight.BOLD),
                    subtitle=ft.Text(f"{song_info['artist']}, {song_info['album_name']} | Score: {score:.3f}", color="white70"),
                    on_click=lambda e, s=song_info: on_song_click(s)
                )
            )
        page.update()

    def find_id(query: str, column: str):
        found = id_information_df[id_information_df[column].str.contains(query, case=False, na=False, regex=False)]
        return found.iloc[0]["id"] if not found.empty else None

    def resolve_unimode_query_id(query):
        for col in ["song", "artist", "album_name"]:
            qid = find_id(query, col)
            if qid: return qid
        return None

    strategies = {
        RetrievalAlgorithms.RANDOM: RandomStrategy(id_information_df["id"].tolist()),
        RetrievalAlgorithms.UNIMODAL: UnimodalStrategy(unimodal_rs),
    }



    def on_song_click(song_data):
        video_url = song_data.get("url", "")

        # Convert standard YouTube link to Embed link
        embed_url = ""
        if "youtube.com" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else video_url.split("/")[-1]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

        # Update the details container with info from the clicked song
        result_container.content = ft.Column([
            ft.Text(f"Title: {song_data['song']}", size=20, weight=ft.FontWeight.BOLD),
            ft.Text(f"Artist: {song_data['artist']}"),
            ft.Text(f"Album: {song_data['album_name']}"),
            ft.Text(f"Score: {song_data['score']:.4f}"),
            ft.Text(f"ID: {song_data['id']}", size=12, color="grey"),
            ft.Divider(height=20, color="transparent"),

            # The Video Player
            # TODO
                ft.Text(f"ID: {song_data['id']}", size=12, color="grey"),
            ], scroll=ft.ScrollMode.AUTO)
        page.update()
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
        on_select=handle_dropdown_menu,
        border_radius=20,
        bgcolor=ft.Colors.DEEP_PURPLE_700,
        border_color=ft.Colors.DEEP_PURPLE_200,
        filled=True,
        fill_color=ft.Colors.DEEP_PURPLE_800,
        trailing_icon=ft.Icon(ft.Icons.ARROW_DROP_DOWN, color=ft.Colors.WHITE),
        expand=True
    )

    search_button = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.SEARCH), ft.Text("Search Now")],
            alignment=ft.MainAxisAlignment.CENTER,
            tight=True,
        ),
        on_click=handle_search_now,
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.DEEP_PURPLE_800,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            side=ft.BorderSide(
                color=ft.Colors.DEEP_PURPLE_200,
                width=1
            ),
            # Padding in ButtonStyle also uses the Padding class
            padding=ft.Padding(20, 20, 20, 20),
        ),
        col={"xs": 8, "sm": 4, "md": 3},
    )

    control_row = ft.ResponsiveRow(
        controls=[
            ft.Container(
                content=search_field,
                alignment=ft.alignment.Alignment.CENTER_RIGHT,
                col={"xs": 12, "md": 6},
            ),
            ft.Container(
                content=dropdown_algorithm,
                alignment=ft.alignment.Alignment.CENTER_LEFT,
                col={"xs": 12, "md": 2},
            ),
            ft.Container(
                content=slider_group,
                alignment=ft.alignment.Alignment.CENTER_LEFT,
                col={"xs": 12, "md": 4},
            )
        ],
        width=float("inf"),
        spacing=20
    )

    result_songs = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        expand=True,
        height=500)

    intermediate_results_container = ft.Container(
        content=result_songs,
        padding=15,
        border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
#        expand=True,
        alignment=ft.Alignment.TOP_LEFT,
        col={"xs": 12, "md": 6}  # xs = small monitor: full width, md = medium = 6 of 12 colums width
    )

    result_container = ft.Container(
        content=ft.Text("Details", color=ft.Colors.WHITE),
        padding=15,
        border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        expand=True,
        alignment=ft.Alignment.TOP_LEFT,
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
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=25,
            expand=True
        )
    )


ft.app(target=main, view=ft.AppView.WEB_BROWSER)  # to run Yamex in PyCharm
# ft.app(target=main) # to run Yamex in terminal go saved directory and use command: flet run --web Yamex.py
# ft.run(main) # alternative for opening a window