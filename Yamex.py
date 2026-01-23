import flet as ft
import flet_video as fv
import pandas as pd

from enum import Enum

from flet import UrlLauncher

from baseline import RandomBaselineRetrievalSystem
from strategies import EarlyFusionStrategy, LateFusionStrategy, RandomStrategy, UnimodalStrategy
from unimodal import UnimodalRetrievalSystem, Evaluator
from early_fusion import EarlyFusionRetrievalSystem
from late_fusion import LateFusionRetrievalSystem

class RetrievalAlgorithms(str, Enum):
    RANDOM = "random"
    UNIMODAL = "unimodal"
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    NEUTRAL_NETWORK = "neutral_network"

class Modality(str, Enum):
    AUDIO = "audio"
    LYRICS = "lyrics"
    VIDEO = "video"
    AUDIO_AUDIO = "audio_audio"
    AUDIO_LYRICS = "audio_lyrics"
    AUDIO_VIDEO = "audio_video"
    LYRICS_LYRICS = "lyrics_lyrics"
    LYRICS_AUDIO = "lyrics_audio"
    LYRICS_VIDEO = "lyrics_video"
    VIDEO_AUDIO = "video_audio"
    VIDEO_LYRICS = "video_lyrics"
    VIDEO_VIDEO = "video_video"
    ALL = "audio_lyrics_video"

# helper directory
MODALITY_MAP = {
    Modality.AUDIO: ["audio"],
    Modality.LYRICS: ["lyrics"],
    Modality.VIDEO: ["video"],
    Modality.AUDIO_LYRICS: ["audio", "lyrics"],
    Modality.AUDIO_VIDEO: ["audio", "video"],
    Modality.LYRICS_VIDEO: ["lyrics", "video"],
    Modality.ALL: ["audio", "lyrics", "video"]
}

DATA_ROOT = "./data"
id_information_df = pd.read_csv(f"{DATA_ROOT}/id_information_mmsr.tsv", sep="\t")
id_genres_df = pd.read_csv(f"{DATA_ROOT}/id_genres_mmsr.tsv", sep="\t")
id_url_df = pd.read_csv(f"{DATA_ROOT}/id_url_mmsr.tsv", sep="\t")

# data frame containing: 'id', 'artist', 'song', 'album_name', 'url', 'genres'
master_df = (id_information_df
               .merge(id_url_df, on="id", how="left")
               .merge(id_genres_df, on="id", how="left")
)
#print(master_df.head(10).to_string())

# convert data frame to dictionary
song_lookup_dict = master_df.set_index("id").to_dict("index")

evaluator = Evaluator(DATA_ROOT)
random_rs = RandomBaselineRetrievalSystem(evaluator, seed=None)
unimodal_rs = UnimodalRetrievalSystem(DATA_ROOT, evaluator)


# Early Fusion Pre-Initialization of all combinations
# it takes long to load at starting program, but it enables a quick search for user
EARLY_FUSION_SYSTEMS = {}
LATE_FUSION_SYSTEMS = {}

MULTIMODAL_MODALITIES = {
    Modality.AUDIO_LYRICS: ["audio", "lyrics"],
    Modality.AUDIO_VIDEO: ["audio", "video"],
    Modality.LYRICS_VIDEO: ["lyrics", "video"],
    Modality.ALL: ["audio", "lyrics", "video"],
}


for modality, modality_list in MULTIMODAL_MODALITIES.items():
    # Early fusion
    try:
        EARLY_FUSION_SYSTEMS[modality] = EarlyFusionRetrievalSystem(
            data_root=DATA_ROOT,
            evaluator=evaluator,
            modalities=modality_list
        )
    except Exception as e:
        print(f"Early fusion init failed for {modality}: {e}")
    # Late fusion
    try:
        LATE_FUSION_SYSTEMS[modality] = LateFusionRetrievalSystem(
            data_root=DATA_ROOT,
            evaluator=evaluator,
            modalities=modality_list
        )
    except Exception as e:
        print(f"Late fusion init failed for {modality}: {e}")


ALGO_ABBREVIATIONS = {
    RetrievalAlgorithms.RANDOM: "rand",
    RetrievalAlgorithms.UNIMODAL: "uni",
    RetrievalAlgorithms.LATE_FUSION: "late f",
    RetrievalAlgorithms.EARLY_FUSION: "early f",
    RetrievalAlgorithms.NEUTRAL_NETWORK: "nn"
}

MODALITY_ABBREVIATIONS = {
    Modality.AUDIO: "a",
    Modality.LYRICS: "l",
    Modality.VIDEO: "v",
    Modality.AUDIO_AUDIO: "a-a",
    Modality.AUDIO_LYRICS: "a-l",
    Modality.AUDIO_VIDEO: "a-v",
    Modality.LYRICS_AUDIO: "l-a",
    Modality.LYRICS_LYRICS: "l-l",
    Modality.LYRICS_VIDEO: "l-v",
    Modality.VIDEO_AUDIO: "v-a",
    Modality.VIDEO_LYRICS: "v-l",
    Modality.VIDEO_VIDEO: "v-v",
    Modality.ALL: "a-l-v"
}


current_slider_value = 10 # default value
current_algorithm = RetrievalAlgorithms.RANDOM
retrieved_results = []
search_history = []

async def main(page: ft.Page):
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
        current_algorithm = dropdown_algorithm.value

        config = {
            RetrievalAlgorithms.RANDOM: [],
            RetrievalAlgorithms.UNIMODAL: [Modality.AUDIO, Modality.LYRICS, Modality.VIDEO],
            RetrievalAlgorithms.EARLY_FUSION: [Modality.AUDIO_LYRICS, Modality.AUDIO_VIDEO,
                                               Modality.LYRICS_VIDEO, Modality.ALL],
            RetrievalAlgorithms.LATE_FUSION: [Modality.AUDIO_LYRICS, Modality.AUDIO_VIDEO,
                                               Modality.LYRICS_VIDEO, Modality.ALL],
            RetrievalAlgorithms.NEUTRAL_NETWORK: [Modality.AUDIO_AUDIO, Modality.AUDIO_LYRICS, Modality.AUDIO_VIDEO,
                                                  Modality.LYRICS_AUDIO, Modality.LYRICS_LYRICS, Modality.LYRICS_VIDEO,
                                                  Modality.VIDEO_AUDIO, Modality.VIDEO_LYRICS, Modality.VIDEO_VIDEO]
        }
        allowed = config.get(current_algorithm, [])
        dropdown_modality.options = [
            ft.dropdown.Option(
                m.value,
                m.value.replace("_", ", ").title())
            for m in allowed
        ]
        if not allowed:
            dropdown_modality.value = None
            dropdown_modality.disabled = True
            dropdown_modality.label = "No Modality Required"
        else:
            dropdown_modality.disabled = False
            dropdown_modality.label = "Modality"
            # Default to first option if current selection is now invalid
            if dropdown_modality.value not in [m.value for m in allowed]:
                dropdown_modality.value = allowed[0].value if allowed else None
        page.update()

    def create_search_field(
        hint_text: str = "Search for a song title, an artist or an album",
        on_submit_callback=None
    ):
        return ft. TextField(
            hint_text=hint_text,
            hint_style=ft.TextStyle(color=ft.Colors.DEEP_PURPLE_800),
            bgcolor=ft.Colors.WHITE,
            border_radius=20,
            color=ft.Colors.BLACK,
            prefix_icon=ft.Icon(ft.Icons.SEARCH_ROUNDED, color=ft.Colors.DEEP_PURPLE_800),
            expand=True,
            on_submit = lambda e: on_submit_callback(e.control.value)
                if on_submit_callback else None
        )

    def log_text(content, weight=ft.FontWeight.NORMAL):
        return ft.Text(
            content,
            size=11,
            font_family="monospace",  # for typewriter look
            weight=weight,
            color=ft.Colors.DEEP_PURPLE_50
        )

    def handle_search_now(e=None):
        query = search_field.value.strip()
        query_id = resolve_unimode_query_id(query)

        result_songs.controls.clear()

        if not query_id:
            result_songs.controls.append(ft.Text("No results found for that query.", color="red"))
            page.update()
            return

        # Select the strategy based on the dropdown value
        selected_algorithm = dropdown_algorithm.value
        selected_modality = dropdown_modality.value
        selected_modality_list = MODALITY_MAP.get(dropdown_modality.value)

        if selected_algorithm == RetrievalAlgorithms.RANDOM:
            strategy = RandomStrategy(random_rs)
            print(selected_algorithm)

        elif selected_algorithm == RetrievalAlgorithms.UNIMODAL:
            strategy = UnimodalStrategy(unimodal_rs, selected_modality)
            print(selected_algorithm, selected_modality_list)

        elif selected_algorithm == RetrievalAlgorithms.EARLY_FUSION:
            ef_rs = EARLY_FUSION_SYSTEMS.get(selected_modality)

            if ef_rs is None:
                result_songs.controls.append(
                    ft.Text("Selected modality not supported for Early Fusion", color="yellow")
                )
                page.update()
                return

            strategy = EarlyFusionStrategy(ef_rs, selected_modality)
            print(selected_algorithm, selected_modality_list)

        elif selected_algorithm == RetrievalAlgorithms.LATE_FUSION:
            lf_rs = LATE_FUSION_SYSTEMS.get(selected_modality)

            if lf_rs is None:
                result_songs.controls.append(
                    ft.Text("Selected modality not supported for Late Fusion", color="yellow")
                )
                page.update()
                return

            strategy = LateFusionStrategy(lf_rs, selected_modality)
            print(selected_algorithm, selected_modality_list)



        else:
            result_songs.controls.append(ft.Text("Not implemented yet", color="yellow"))
            page.update()
            return

        # execute search
        ids, raw_metrics, scores = strategy.search(query_id, current_slider_value)
        print(f"DEBUG: scores: {scores}")
        print(f"DEBUG: metrics: {raw_metrics}")

        # 1. Store cleaned metrics in a dictionary (from numpy type to standard)
        current_metrics = {
            "algorithm": dropdown_algorithm.value,
            "modality": dropdown_modality.value,
            "precision": float(raw_metrics.get(f"Precision@{current_slider_value}", 0.0)),
            "recall": float(raw_metrics.get(f"Recall@{current_slider_value}", 0.0)),
            "mrr": float(raw_metrics.get(f"MRR@{current_slider_value}", 0.0)),
            "ndcg": float(raw_metrics.get(f"nDCG@{current_slider_value}", 0.0))
        }
        print(current_metrics)
        # 2. Update the UI using your dictionary
        precision_text.value = f"Precision@{current_slider_value}: {current_metrics['precision']:.4f}  "
        recall_text.value = f"  Recall@{current_slider_value}: {current_metrics['recall']:.4f}  "
        mmr_text.value = f"  MRR@{current_slider_value}:  {current_metrics['mrr']:.4f}  "
        ndcg_text.value = f"  nDCG@{current_slider_value}:  {current_metrics['ndcg']:.4f}"

        metrics_display.visible=True
        search_history.append(current_metrics)

        algo_abbr = ALGO_ABBREVIATIONS.get(dropdown_algorithm.value, "-"),
        modality_abbr = MODALITY_ABBREVIATIONS.get(dropdown_modality.value, "-")

        history_column.controls.insert(0, ft.Container(
            padding=ft.Padding.only(bottom=5),
            content=ft.Row([
                ft.Container(content=log_text(algo_abbr), width=50),
                ft.Container(content=log_text(modality_abbr), width=40),
                ft.Container(content=log_text(f"{current_metrics['precision']:.4f}"), width=45),
                ft.Container(content=log_text(f"{current_metrics['recall']:.4f}"), width=45),
                ft.Container(content=log_text(f"{current_metrics['mrr']:.4f}"), width=45),
                ft.Container(content=log_text(f"{current_metrics['ndcg']:.4f}"), width=45)
            ], spacing=10)
        ))

        history_log_container.visible=True
        page.update()

        # update UI
        retrieved_results.clear()
        for i, (ret_id, score) in enumerate(zip(ids, scores)):
            res = song_lookup_dict.get(ret_id)
            if not res:
                continue

            # Convert NumPy float64 to Python float
            display_score = float(score)

            song_info = {
                "index": i + 1,
                "id": ret_id,
                "song": res["song"],
                "artist": res["artist"],
                "album_name": res["album_name"],
                "url": res["url"],
                "genres": res.get("genre", "N/A"),
                "score": display_score,
            }
            retrieved_results.append(song_info)

            result_songs.controls.append(
                ft.ListTile(
                    leading=ft.Text(f"[{song_info['index']}]", color="white70", size=14),
                    title=ft.Text(f"{song_info['song']}", color="white", weight=ft.FontWeight.BOLD),
                    subtitle=ft.Text(f"{song_info['artist']}, {song_info['album_name']}", color="white70"),
                    on_click=lambda e, s=song_info: on_song_click(s),
                    trailing=ft.Text(f" score: {display_score:.4f}", color="white70", size=14),
                )
            )

        page.update()

    def find_id(query: str, column: str):
        found = id_information_df[id_information_df[column].str.contains(query, case=False, na=False, regex=False)]
        return found.iloc[0]["id"] if not found.empty else None

    def resolve_unimode_query_id(query):
        query = query.lower()
        # Search across song, artist, and album_name simultaneously
        mask = (
                master_df["song"].str.contains(query, case=False, na=False) |
                master_df["artist"].str.contains(query, case=False, na=False) |
                master_df["album_name"].str.contains(query, case=False, na=False)
        )
        found = master_df[mask]
        return found.iloc[0]["id"] if not found.empty else None

    def on_song_click(song_data):
        video_url = song_data.get("url", "")
        genres_raw = song_data.get("genres", "No genres listed")
        clean_genres = str(genres_raw).strip("[]").replace("'", "").replace('"', '')

        genre_chips = ft.Row(
            wrap=True,
            spacing=5,
            controls=[
                ft.Container(
                    content=ft.Text(g.strip(), color=ft.Colors.WHITE),
                    bgcolor=ft.Colors.DEEP_PURPLE_700,
                    padding=ft.Padding(10,5,10,5),
                    border_radius=15,
                    border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
               ) for g in str(clean_genres).split(",") if g.strip() and g.strip().lower() != "nan"
            ],
        )

       # 1. Transform the URL (Crucial for iframe security)
        if "youtube.com" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else video_url.split("/")[-1]
            embed_url = f"https://www.youtube.com/embed/{video_id}"
        else:
            embed_url = video_url

        video_player = fv.Video(
            expand=True,
            playlist=[fv.VideoMedia(embed_url)],
            aspect_ratio=16 / 9,
            autoplay=False,
            # essential for Linux:
            show_controls=True,
        )

        async def handle_open_link(e):
            await UrlLauncher().launch_url(video_url)

        # Update the details container with info from the clicked song
        result_container.content = ft.Column([
            ft.Text(f"Title: {song_data['song']}", size=20, weight=ft.FontWeight.BOLD),
            ft.Text(f"Artist: {song_data['artist']}"),
            ft.Text(f"Album: {song_data['album_name']}"),
#            ft.Text(f"ID: {song_data['id']}", size=12, color="grey"),
            ft.Divider(height=1, color="transparent"),
            ft.Text("Genres:" ),
            genre_chips,
            ft.Divider(height=1, color="transparent"),
            # The Video Player
            ft.Container(
                content=video_player,
                border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
                border_radius=10,
                padding=10,
            ),
            # Add a direct link button as a backup
            ft.Button(
                content=ft.Text("Open Video in New Tab"),
                icon=ft.Icons.OPEN_IN_NEW,
                on_click=handle_open_link
            ),
#            ft.Text(f"URL: {song_data['url']}", size=12, color="grey"),
            ], scroll=ft.ScrollMode.AUTO)
        page.update()

    search_field = create_search_field(on_submit_callback=handle_search_now)

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
        width=float("inf"),
#        width=300,
        value=RetrievalAlgorithms.RANDOM.value,  # start value
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

    dropdown_modality = ft.Dropdown(
        label="Modality",
        label_style=ft.TextStyle(color=ft.Colors.WHITE),
        text_style=ft.TextStyle(color=ft.Colors.WHITE),
        width=float("inf"),
#        width=300,
        value=None,  # start value
        options=[],
        disabled=True,
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
            padding=ft.Padding(20, 20, 20, 20),
        ),
        col={"xs": 8, "sm": 4, "md": 3},
    )

    control_grid = ft.ResponsiveRow(
        controls=[
            # first row
            ft.Container(
                content=search_field,
                alignment=ft.alignment.Alignment.CENTER_RIGHT,
                col={"xs": 12, "md": 6},
            ),
            ft.Container(
                content=ft.Row([dropdown_algorithm]),
                alignment=ft.alignment.Alignment.CENTER_LEFT,
                col={"xs": 12, "md": 3},
            ),
            # second row
            ft.Container(
                content=slider_group,
                alignment=ft.alignment.Alignment.CENTER_LEFT,
                col={"xs": 12, "md": 6},
            ),
            ft.Container(
                content=ft.Row([dropdown_modality]),
                alignment=ft.alignment.Alignment.CENTER_LEFT,
                col={"xs": 12, "md": 3},
            ),
            ft.Container(
                content=search_button,
                alignment=ft.alignment.Alignment.CENTER_LEFT,
                col={"xs": 12, "md": 3},
            )
        ],
        width=float("inf"),
        spacing=20
    )

    precision_text = ft.Text("Precision: --", color=ft.Colors.WHITE)
    recall_text = ft.Text("Recall: --", color=ft.Colors.WHITE)
    mmr_text = ft.Text("MRR: --", color=ft.Colors.WHITE)
    ndcg_text = ft.Text("nDCG: --", color=ft.Colors.WHITE)

    metrics_display = ft.Container(
        content=ft.Row(
            [precision_text, recall_text, mmr_text, ndcg_text],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        bgcolor=ft.Colors.DEEP_PURPLE_800,
        visible=False  # hidden by default
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
        col={"xs": 12, "md": 4}  # xs = small monitor: full width, md = medium = 4 of 12 colums width
    )

    result_container = ft.Container(
        content=ft.Text("Details", color=ft.Colors.WHITE),
        padding=15,
        border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        expand=True,
        alignment=ft.Alignment.TOP_LEFT,
        col={"xs": 12, "md": 5}
    )

    history_column = ft.Column()
    history_log_container = ft.Container(
        content=ft.Column([
            ft.Text("Comparison Log: ", weight="bold"),
            ft.Divider(color=ft.Colors.DEEP_PURPLE_200, height=1),
            ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            ft.Row([
                            # header
                            ft.Container(content=log_text("ALGO", ft.FontWeight.BOLD), width=50),
                            ft.Container(content=log_text("MOD", ft.FontWeight.BOLD), width=40),
                            ft.Container(content=log_text("PREC", ft.FontWeight.BOLD), width=45),
                            ft.Container(content=log_text("RECL", ft.FontWeight.BOLD), width=45),
                            ft.Container(content=log_text("MRR", ft.FontWeight.BOLD), width=45),
                            ft.Container(content=log_text("nDCG", ft.FontWeight.BOLD), width=45),
                        ], spacing=10),
                        ft.Divider(color=ft.Colors.DEEP_PURPLE_200, height=1),
                        # data rows
                        history_column,
                        ],
                        scroll=ft.ScrollMode.ALWAYS, expand=True  # vertical scroll
                    )
                ],
                scroll=ft.ScrollMode.ALWAYS, expand=True  # horizontal scroll
            )
        ]),
        bgcolor=ft.Colors.DEEP_PURPLE_900,
        padding = 15,
        border = ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius = 20,
        col={"xs": 12, "md": 3},
        visible=False
    )

    result_row = ft.ResponsiveRow(
        controls=[
            intermediate_results_container,
            result_container,
            history_log_container
        ],
        spacing=25,
        alignment=ft.MainAxisAlignment.START
    )

    page.add(
        ft.Column(
            controls=[
                title,
                control_grid,
                metrics_display,
                ft.Text("Top results:"),
                result_row
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=25,
            expand=True
        )
    )
    handle_dropdown_menu(None)

ft.run(main)  # open YAMEx in a separate window
#ft.run(main, view=ft.AppView.WEB_BROWSER) # opens YAMEx in browser