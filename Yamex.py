import flet as ft
import flet_video as fv
import pandas as pd
import random as rlib

from flet import UrlLauncher

from baseline import RandomBaselineRetrievalSystem
from strategies import *
from unimodal import UnimodalRetrievalSystem, Evaluator
from early_fusion import EarlyFusionRetrievalSystem
from late_fusion import LateFusionRetrievalSystem
from nn_based import NeuralNetworkBasedRetrievalSystem


DATA_ROOT = "./data"
NN_DATA_ROOT = "./data_nn/NN_pretrained_models_and_features/"
FIXED_CONTAINER_HEIGHT = 620
id_information_df = pd.read_csv(f"{DATA_ROOT}/id_information_mmsr.tsv", sep="\t")
id_genres_df = pd.read_csv(f"{DATA_ROOT}/id_genres_mmsr.tsv", sep="\t")
id_url_df = pd.read_csv(f"{DATA_ROOT}/id_url_mmsr.tsv", sep="\t")

# data frame containing: 'id', 'artist', 'song', 'album_name', 'url', 'genres'
master_df = (id_information_df
               .merge(id_url_df, on="id", how="left")
               .merge(id_genres_df, on="id", how="left")
)

# convert data frame to dictionary
song_lookup_dict = master_df.set_index("id").to_dict("index")

evaluator = Evaluator(DATA_ROOT)
random_rs = RandomBaselineRetrievalSystem(evaluator, seed=None)
unimodal_rs = UnimodalRetrievalSystem(DATA_ROOT, evaluator)

# Early fusion, late fusion and neural-network based pre-Initialization of all combinations
# it takes long to load at starting program (app. 45-90 sec), but it enables a quick search for user
EARLY_FUSION_SYSTEMS = {}
LATE_FUSION_SYSTEMS = {}
NN_SYSTEMS = {}

FUSION_MODALITIES = {
    Modality.AUDIO_LYRICS: ["audio", "lyrics"],
    Modality.AUDIO_VIDEO: ["audio", "video"],
    Modality.LYRICS_VIDEO: ["lyrics", "video"],
    Modality.ALL: ["audio", "lyrics", "video"],
}


for modality, modality_list in FUSION_MODALITIES.items():
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


NN_MODALITIES = [
    Modality.AUDIO_AUDIO, Modality.AUDIO_LYRICS, Modality.AUDIO_VIDEO,
    Modality.LYRICS_AUDIO, Modality.LYRICS_LYRICS, Modality.LYRICS_VIDEO,
    Modality.VIDEO_AUDIO, Modality.VIDEO_LYRICS, Modality.VIDEO_VIDEO
]

for modality in NN_MODALITIES:
    try:
        query_mod, result_mod = MODALITY_MAP[modality]  # to separate in input and output modality
        NN_SYSTEMS[modality] = NeuralNetworkBasedRetrievalSystem(
            data_root=NN_DATA_ROOT,
            evaluator=evaluator,
            query_modality=query_mod,
            result_modality=result_mod
        )
    except Exception as e:
        print(f"Neural-Network based retrieval system init failed for {modality}: {e}")


ALGO_ABBREVIATIONS = {
    RetrievalAlgorithms.RANDOM: "rand",
    RetrievalAlgorithms.UNIMODAL: "uni",
    RetrievalAlgorithms.LATE_FUSION: "late f",
    RetrievalAlgorithms.EARLY_FUSION: "early f",
    RetrievalAlgorithms.NEURAL_NETWORK: "nn"
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


current_slider_value = 10  # default value
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
            ft.TextSpan(" - Yet Another Music Explorer", ft.TextStyle(size=20)),
        ],
        text_align=ft.TextAlign.CENTER
    )

    search_field = ft.TextField(
        hint_text="Search for a song title, artist, or album...",
            hint_style=ft.TextStyle(color=ft.Colors.DEEP_PURPLE_900),
            bgcolor=ft.Colors.DEEP_PURPLE_50,
            border=ft.InputBorder.OUTLINE,
            border_radius=20,
            border_width=1,
            border_color = ft.Colors.DEEP_PURPLE_200,
            color=ft.Colors.BLACK,
            prefix_icon=ft.Icon(ft.Icons.SEARCH_ROUNDED, color=ft.Colors.DEEP_PURPLE_900),
            expand=True,
            hover_color=ft.Colors.WHITE,
    )

    def on_search_change(e):
        # check if search_field.border_width was set before
        if search_field.border_width is not None and search_field.border_width > 1:
            reset_search_field_style()

    search_field.on_change = on_search_change
    search_field.on_submit = lambda e: handle_search_now()

    # text element to show query details
    query_info_text = ft.Text("", color=ft.Colors.DEEP_PURPLE_100, weight=ft.FontWeight.W_500)

    dropdown_matching_songs = ft.Dropdown(
        label="Matching songs",
        tooltip="If the random algorithm has not been selected, all songs found with the query will be displayed here.",
        label_style=ft.TextStyle(color=ft.Colors.WHITE),
        text_style=ft.TextStyle(color=ft.Colors.WHITE),
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.DEEP_PURPLE_700,
        border_color=ft.Colors.DEEP_PURPLE_200,
        filled=True,
        fill_color=ft.Colors.DEEP_PURPLE_800,
        border_radius=20,
        width=600,
        visible=True,
        on_select=lambda e: execute_retrieval(dropdown_matching_songs.value),
    )

    def clear_search_results(e):
        reset_ui_displays()
        # switch to and fire random search
        dropdown_algorithm.value = RetrievalAlgorithms.RANDOM.value
        handle_dropdown_algo_and_mod(None)


    clear_search_results_button = ft.IconButton(
        icon=ft.Icons.DELETE_SWEEP_ROUNDED,
        icon_color=ft.Colors.DEEP_PURPLE_200,
        tooltip="Clears search results and triggers a random search",
        on_click=clear_search_results,
    )

    query_info_display = ft.Container(
        content=ft.Row(
            [
                query_info_icon := ft.Icon(ft.Icons.INFO_OUTLINE, color=ft.Colors.DEEP_PURPLE_200, size=20, visible=False),
                dropdown_matching_songs,
                clear_search_results_button
            ],
            alignment=ft.MainAxisAlignment.CENTER,  # horizontal centering
            vertical_alignment=ft.CrossAxisAlignment.CENTER,  # vertical centering
        ),
        bgcolor=ft.Colors.TRANSPARENT,  # tansparent if no infos are shown
        height=70,  # reserves space that info later needs
        border_radius=10,
    )

    slider_label = ft.Text(
        f"Number of results: {current_slider_value}",
        color=ft.Colors.WHITE,
        width=180,  # fixed width of label so slider does not get narrower with multi-digit number
        no_wrap=True
    )

    def handle_slider(e: ft.ControlEvent):#
        global current_slider_value
        current_slider_value = int(e.control.value)
        slider_label.value = f" Number of results: {current_slider_value}"
        page.update()

    def generate_random_song_id():
        return rlib.choice(list(song_lookup_dict.keys()))  # pick a random id for the evaluator

    def handle_dropdown_algo_and_mod(e):
        global current_algorithm
        current_algorithm = dropdown_algorithm.value

        if current_algorithm == RetrievalAlgorithms.RANDOM.value:
            reset_search_field_style()
            dropdown_matching_songs.label = "Selecting a song is not possible with random algorithm"
            dropdown_matching_songs.value = ""
            dropdown_matching_songs.options = []
            query_info_icon.visible = True
            dropdown_matching_songs.disabled = True
            page.update()
        else:
            dropdown_matching_songs.disabled = False
            dropdown_matching_songs.label = "Matching songs"

        config = {
            RetrievalAlgorithms.RANDOM: [],
            RetrievalAlgorithms.UNIMODAL: [Modality.AUDIO, Modality.LYRICS, Modality.VIDEO],
            RetrievalAlgorithms.EARLY_FUSION: [Modality.AUDIO_LYRICS, Modality.AUDIO_VIDEO,
                                               Modality.LYRICS_VIDEO, Modality.ALL],
            RetrievalAlgorithms.LATE_FUSION: [Modality.AUDIO_LYRICS, Modality.AUDIO_VIDEO,
                                               Modality.LYRICS_VIDEO, Modality.ALL],
            RetrievalAlgorithms.NEURAL_NETWORK: [Modality.AUDIO_AUDIO, Modality.AUDIO_LYRICS, Modality.AUDIO_VIDEO,
                                                 Modality.LYRICS_AUDIO, Modality.LYRICS_LYRICS, Modality.LYRICS_VIDEO,
                                                 Modality.VIDEO_AUDIO, Modality.VIDEO_LYRICS, Modality.VIDEO_VIDEO]
        }
        allowed = config.get(current_algorithm, [])
        dropdown_modality.options = [
            ft.dropdown.Option(
                m.value,
                m.value.replace("_", ", ").title())  # replace _ from enum with , for a more appealing look
            for m in allowed
        ]
        if not allowed:
            dropdown_modality.value = None
            dropdown_modality.disabled = True
            dropdown_modality.label = "No Modality Required"
        else:
            dropdown_modality.disabled = False
            dropdown_modality.label = "Modality"
            # Default: select first option if current selection is now invalid
            if dropdown_modality.value not in [m.value for m in allowed]:
                dropdown_modality.value = allowed[0].value if allowed else None

        # Auto-Retrigger if user changes algorithm or modality after selecting a song
        if current_algorithm == RetrievalAlgorithms.RANDOM.value:
            execute_retrieval(generate_random_song_id())   # pick a random id for the evaluator
        elif dropdown_matching_songs.value:  # fire new search after changing algorithm or modality
            execute_retrieval(dropdown_matching_songs.value)
        page.update()

    def log_text(content, weight=ft.FontWeight.NORMAL):
        return ft.Text(
            content,
            size=11,
            font_family="monospace",  # for typewriter look and same space usage
            weight=weight,
            color=ft.Colors.DEEP_PURPLE_50
        )

    def reset_search_field_style():
        search_field.value = ""  # if another algorithm is selected, clear search field
        search_field.border_color = ft.Colors.DEEP_PURPLE_200  # usage of standard color again
        search_field.border_width = 1  # reset standard border
        search_field.bgcolor = ft.Colors.DEEP_PURPLE_50  # reset to standard background color
        search_field.update()

    def reset_ui_displays():
        # clear query infos
        query_info_text.value = ""
        query_info_text.spans = []
        query_info_icon.visible = False
        query_info_display.bgcolor = ft.Colors.TRANSPARENT

        result_songs.controls.clear()
        results_title.value = "Enter a search query or use random algorithm"
        detail_container.content = ""

        # clear metrics
        for t in [precision_text, recall_text, mmr_text, ndcg_text]:
            t.color = ft.Colors.TRANSPARENT
        metrics_display.bgcolor = ft.Colors.TRANSPARENT

        page.update()

    def handle_search_now(e=None):
        selected_algorithm = dropdown_algorithm.value

        if selected_algorithm == RetrievalAlgorithms.RANDOM.value:
            search_field.value = ""
            search_field.update()
            execute_retrieval(generate_random_song_id())  # random query_id for Evaluator
            return

        query = search_field.value.strip()

        # Validation of user input - if not valid drag attention of user to input field
        if selected_algorithm != RetrievalAlgorithms.RANDOM.value and not query:
            reset_ui_displays()
            search_field.bgcolor = ft.Colors.AMBER_50
            search_field.border_color = ft.Colors.AMBER_600  # to drag attention to the input field
            search_field.border_width = 4
            search_field.update()
            return

        # Match finding
        matches = resolve_query_id(query)

        # Get all songs that matches the query
        if not matches and dropdown_algorithm.value != RetrievalAlgorithms.RANDOM.value:
            reset_ui_displays()
            result_songs.controls.clear()
            result_songs.controls.append(ft.Text("Sorry, no results were found for your query.",
                                                 color=ft.Colors.AMBER_400,
                                                 weight=ft.FontWeight.BOLD,
                                                 size=20
                                                 ))
            page.update()
            return

        # Update dropdown options (fill options with matching songs)
        dropdown_matching_songs.options = [
            ft.dropdown.Option(m["id"], m["display"]) for m in matches
        ]

        if matches:
            dropdown_matching_songs.value = matches[0]["id"]  # Default selection
            dropdown_matching_songs.visible = True
            query_info_display.bgcolor = ft.Colors.DEEP_PURPLE_800
            execute_retrieval(dropdown_matching_songs.value)  # as default trigger retrieval with first matching song
        else:
            execute_retrieval(generate_random_song_id())  # pick a random id for the evaluator


    def execute_retrieval(selected_id):
        # Select the strategy based on the dropdown value
        selected_algorithm = dropdown_algorithm.value
        selected_modality = dropdown_modality.value

        result_songs.controls.clear()

        # Reset of info board (no content, but keeps space)
        query_info_text.value = ""
        query_info_text.spans = []
        query_info_icon.visible = False
        query_info_display.bgcolor = ft.Colors.TRANSPARENT

        if selected_id and dropdown_algorithm.value != RetrievalAlgorithms.RANDOM.value:
            # activate only if not random algorithm is selected
            q_details = song_lookup_dict.get(selected_id, {})
            on_song_click(q_details, is_query=True)
            query_info_text.spans = [
                ft.TextSpan(f"Query: ", ft.TextStyle(weight=ft.FontWeight.BOLD, color=ft.Colors.DEEP_PURPLE_200)),
                ft.TextSpan(f"{q_details.get('song', 'Unknown')} "
                            f"by {q_details.get('artist', 'Unknown')} "
                            f"from {q_details.get('album', 'Unknown')} "),
            ]
            query_info_icon.visible = True
            query_info_display.bgcolor = ft.Colors.DEEP_PURPLE_900
        elif selected_algorithm == RetrievalAlgorithms.RANDOM.value:
            query_info_text.value = "Random Baseline Search"
            query_info_icon.visible = False
            query_info_display.bgcolor = ft.Colors.DEEP_PURPLE_900

        page.update()

        if selected_algorithm == RetrievalAlgorithms.RANDOM.value:
            strategy = RandomStrategy(random_rs)

        elif selected_algorithm == RetrievalAlgorithms.UNIMODAL.value:
            strategy = UnimodalStrategy(unimodal_rs, selected_modality)

        elif selected_algorithm == RetrievalAlgorithms.EARLY_FUSION.value:
            ef_rs = EARLY_FUSION_SYSTEMS.get(selected_modality)
            if ef_rs is None:
                result_songs.controls.append(
                    ft.Text("Early Fusion is currently not loaded and is unavailable", color="yellow")
                )
                page.update()
                return
            strategy = EarlyFusionStrategy(ef_rs, selected_modality)

        elif selected_algorithm == RetrievalAlgorithms.LATE_FUSION.value:
            lf_rs = LATE_FUSION_SYSTEMS.get(selected_modality)
            if lf_rs is None:
                result_songs.controls.append(
                    ft.Text("Late Fusion is currently not loaded and is unavailable", color="yellow")
                )
                page.update()
                return
            strategy = LateFusionStrategy(lf_rs, selected_modality)

        elif selected_algorithm == RetrievalAlgorithms.NEURAL_NETWORK.value:
            nn_rs = NN_SYSTEMS.get(selected_modality)
            if nn_rs is None:
                result_songs.controls.append(
                    ft.Text("Neural Network is currently not loaded and is unavailable.", color="yellow")
                )
                page.update()
                return
            strategy = NeuralNetworkStrategy(nn_rs, selected_modality)
        else:
            result_songs.controls.append(ft.Text("Not implemented yet", color="yellow"))
            page.update()
            return

        # execute search
        ids, raw_metrics, scores = strategy.search(selected_id, current_slider_value)

        if ids:
            if selected_algorithm == RetrievalAlgorithms.RANDOM.value:
                results_title.value = f"RANDOM SONGS TO EXPLORE:"
            else:
                results_title.value = f"TOP RESULTS:"
        else:
            results_title.value = f"Sorry, no results were found for your query."



        page.update()

        # Store cleaned metrics in a dictionary (from numpy type to standard)
        current_metrics = {
            "algorithm": dropdown_algorithm.value,
            "modality": dropdown_modality.value,
            "precision": float(raw_metrics.get(f"Precision@{current_slider_value}") or 0.0),
            "recall": float(raw_metrics.get(f"Recall@{current_slider_value}") or 0.0),
            "mrr": float(raw_metrics.get(f"MRR@{current_slider_value}") or 0.0),
            "ndcg": float(raw_metrics.get(f"nDCG@{current_slider_value}") or 0.0)
        }
        # Update the UI with the metrics values
        precision_text.value = f"Precision@{current_slider_value:<3}: {current_metrics['precision']:.4f}"
        precision_text.color = ft.Colors.WHITE
        recall_text.value = f"Recall@{current_slider_value:<3}: {current_metrics['recall']:.4f}"
        recall_text.color = ft.Colors.WHITE
        mmr_text.value = f"MRR@{current_slider_value:<3}: {current_metrics['mrr']:.4f}"
        mmr_text.color = ft.Colors.WHITE
        ndcg_text.value = f"nDCG@{current_slider_value:<3}: {current_metrics['ndcg']:.4f}"
        ndcg_text.color = ft.Colors.WHITE
        metrics_display.bgcolor = ft.Colors.DEEP_PURPLE_800

        search_history.append(current_metrics)

        algo_abbr = ALGO_ABBREVIATIONS.get(dropdown_algorithm.value, "-")
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

            display_score = float(score)  # Convert NumPy float64 to Python float

            song_info = {
                "index": i + 1,
                "id": ret_id,
                "song": res["song"],
                "artist": res["artist"],
                "album_name": res["album_name"],
                "url": res["url"],
                "genre": res.get("genre", "N/A"),
                "score": display_score,
            }
            retrieved_results.append(song_info)

            result_songs.controls.append(
                ft.ListTile(
                    leading=ft.Text(f"[{song_info['index']}]", color="white70", size=14),
                    title=ft.Text(f'"{song_info["song"]}"', color="white", weight=ft.FontWeight.BOLD),
                    subtitle=ft.Text(f"{song_info['artist']}  ({song_info['album_name']})", color="white70"),
                    on_click=lambda e, s=song_info: on_song_click(s),  # it will automatically use is_query=False
                    trailing=ft.Text(f" score: {display_score:.4f}", color="white70", size=14),
                )
            )

        page.update()


    def resolve_query_id(query):
        query = query.lower()
        # Search across song, artist, and album_name in that order with given query
        mask = (
                master_df["song"].str.contains(query, case=False, na=False) |
                master_df["artist"].str.contains(query, case=False, na=False) |
                master_df["album_name"].str.contains(query, case=False, na=False)
        )
        found = master_df[mask]

        results = []
        for _, row in found.iterrows():
            display_matches_text = f'"{row["song"]}" - {row["artist"]} ({row["album_name"]})'
            results.append({
                "id": row["id"],
                "display": display_matches_text
            })
        return results

    video_player = fv.Video(
        expand=True,
        aspect_ratio=16 / 9,
        autoplay=False,
        show_controls=True  # essential for Linux
    )

    def on_song_click(song_data, is_query=False):  # Flag: is_query
        video_url = song_data.get("url", "")
        genres_raw = song_data.get("genre", "No genres listed")
        clean_genres = str(genres_raw).strip("[]").replace("'", "").replace('"', '')

        genre_chips = ft.Row(
            wrap=True,
            spacing=5,
            controls=[
                ft.Container(
                    content=ft.Text(g.strip(), color=ft.Colors.WHITE),
                    bgcolor=ft.Colors.DEEP_PURPLE_700,
                    padding=ft.Padding(10,3,10,3),
                    border_radius=15,
                    border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
               ) for g in str(clean_genres).split(",") if g.strip() and g.strip().lower() != "nan"
            ],
        )

       # Transform the URL (Crucial for iframe security)
        if "youtube.com" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else video_url.split("/")[-1]
            embed_url = f"https://www.youtube.com/embed/{video_id}"
        else:
            embed_url = video_url

        video_player.playlist = [fv.VideoMedia(embed_url)]

        async def handle_open_link(e):
            await UrlLauncher().launch_url(video_url)

        # Update the details container with info from the clicked song
        detail_container.content = ft.Column([
            ft.Text("QUERY SONG" if is_query else "SELECTED RESULT SONG",
                    size=12, color=ft.Colors.DEEP_PURPLE_200, weight=ft.FontWeight.BOLD),
            ft.Text(f'Title: "{song_data["song"]}"', size=20, weight=ft.FontWeight.BOLD),
            ft.Column([
                ft.Text(f"Artist: {song_data['artist']}"),
                ft.Text(f"Album: {song_data['album_name']}"),
            ], spacing=2 ),
            ft.Text("Genres:" ),
            genre_chips,
            # Add a direct link button as a backup
            ft.Button(
                content=ft.Text(f"Open Video in New Tab: {song_data['url']}", size=12, color=ft.Colors.DEEP_PURPLE_900),
                icon=ft.Icons.OPEN_IN_NEW,
                icon_color=ft.Colors.DEEP_PURPLE_900,
                on_click=handle_open_link,
                bgcolor=ft.Colors.DEEP_PURPLE_100,
            ),

            # The Video Player
            ft.Container(
                content=video_player,
                border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
                border_radius=10,
                padding=10,
                align=ft.Alignment.CENTER,
                height=350  # max px height of video container
            )
            ], scroll=ft.ScrollMode.AUTO)
        page.update()

    results_slider = ft.Slider(
        min=1,
        max=200,
        divisions=40,  # for step size of 5
        value=current_slider_value,
        on_change=handle_slider,
        expand=True
    )

    slider_group = ft.Row(
        controls=[
            slider_label,
            results_slider
        ],
    )

    dropdown_algorithm = ft.Dropdown(
        label="Algorithm",
        tooltip="Select the algorithm you want to use",
        label_style=ft.TextStyle(color=ft.Colors.WHITE),
        text_style=ft.TextStyle(color=ft.Colors.WHITE),
        value=RetrievalAlgorithms.RANDOM.value,  # start value
        options=[
            ft.dropdown.Option(RetrievalAlgorithms.RANDOM.value, "Random baseline"),
            ft.dropdown.Option(RetrievalAlgorithms.UNIMODAL.value, "Unimodal"),
            ft.dropdown.Option(RetrievalAlgorithms.EARLY_FUSION.value, "Multimodal - Early fusion"),
            ft.dropdown.Option(RetrievalAlgorithms.LATE_FUSION.value, "Multimodal - Late fusion"),
            ft.dropdown.Option(RetrievalAlgorithms.NEURAL_NETWORK.value, "Neural-Network based")
        ],
        on_select=handle_dropdown_algo_and_mod,
        border_radius=20,
        bgcolor=ft.Colors.DEEP_PURPLE_700,
        border_color=ft.Colors.DEEP_PURPLE_200,
        filled=True,
        fill_color=ft.Colors.DEEP_PURPLE_800,
        trailing_icon=ft.Icon(ft.Icons.ARROW_DROP_DOWN, color=ft.Colors.WHITE),
        width = 260
    )

    dropdown_modality = ft.Dropdown(
        label="Modality",
        tooltip="Select the modality for the chosen algorithm",
        label_style=ft.TextStyle(color=ft.Colors.WHITE),
        text_style=ft.TextStyle(color=ft.Colors.WHITE),
        value=None,  # start value
        options=[],
        disabled=True,
        on_select=handle_dropdown_algo_and_mod,
        border_radius=20,
        bgcolor=ft.Colors.DEEP_PURPLE_700,
        border_color=ft.Colors.DEEP_PURPLE_200,
        filled=True,
        fill_color=ft.Colors.DEEP_PURPLE_800,
        trailing_icon=ft.Icon(ft.Icons.ARROW_DROP_DOWN, color=ft.Colors.WHITE),
        width = 260
    )

    search_button = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.SEARCH, color=ft.Colors.DEEP_PURPLE_900),
             ft.Text("Search Now",
                     color=ft.Colors.DEEP_PURPLE_900,
                     weight=ft.FontWeight.BOLD,
                     size=18)],
            alignment=ft.MainAxisAlignment.CENTER,
            tight=True,
        ),
        on_click=handle_search_now,
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=20),
            side=ft.BorderSide(
                color=ft.Colors.DEEP_PURPLE_800,
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
                col={"xs": 12, "md": 6},
            ),
            ft.Container(
                content=ft.Row([dropdown_algorithm], expand=True),
                col={"xs": 12, "md": 2},
                alignment=ft.Alignment.CENTER
            ),
            ft.Container(  # placeholder for symmetry
                col={"xs": 12, "md": 4},
            ),
            # second row
            ft.Container(
                content=slider_group,
                col={"xs": 12, "md": 6},
            ),
            ft.Container(
                content=dropdown_modality,
                col={"xs": 12, "md": 1.8},
            ),
            ft.Container(
                content=search_button,
                col={"xs": 12, "md": 3},
                alignment=ft.Alignment.CENTER_LEFT
            )
        ],
        spacing=20
    )

    # text elements to show evaluation matrics
    precision_text = ft.Text(font_family="monospace", size=13)
    recall_text = ft.Text(font_family="monospace", size=13)
    mmr_text = ft.Text(font_family="monospace", size=13)
    ndcg_text = ft.Text(font_family="monospace", size=13)

    metrics_display = ft.Container(
        content=ft.Row(
            [precision_text, recall_text, mmr_text, ndcg_text],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=30
        ),
        bgcolor=ft.Colors.TRANSPARENT,  # transparent if no matrics shown to keep space
        border_radius=10,
        height=30
    )

    result_songs = ft.Column(
        scroll=ft.ScrollMode.ALWAYS,
        expand=True,
    )

    results_title = ft.Text(
        value = "Top Results shown after search...",
        size=12, color=ft.Colors.DEEP_PURPLE_200, weight=ft.FontWeight.BOLD)

    results_container = ft.Container(
        content=ft.Column([
            results_title,
            ft.Container(content=result_songs, expand=True),
            ]),
        padding=15,
        border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        expand=True,
        alignment=ft.Alignment.TOP_LEFT,
        height=FIXED_CONTAINER_HEIGHT,
        col={"xs": 12, "md": 4}  # xs = small monitor: full width, md = medium = 4 of 12 colums width
    )

    detail_container = ft.Container(
        content=ft.Text("FOR DETAILS CLICK ON A SONG ON THE LEFT OR SELECT ONE FROM THE DROPDOWN FOR MATCHING SONGS :-)",
                        size=12, color=ft.Colors.DEEP_PURPLE_200, weight=ft.FontWeight.BOLD),
        padding=15,
        border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        expand=True,
        alignment=ft.Alignment.TOP_LEFT,
        height=FIXED_CONTAINER_HEIGHT,
        col={"xs": 12, "md": 5.5}
    )

    def clear_history_log(e):
        history_column.controls.clear()
        search_history.clear()
        page.update()

    history_column = ft.Column()

    history_log_container = ft.Container(
        content=ft.Column(
            controls=[
                # Header Row
                ft.Row([
                    ft.Text("Comparison Log: ", weight="bold", expand=True),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_SWEEP_ROUNDED,
                        icon_color=ft.Colors.DEEP_PURPLE_200,
                        tooltip="Clear Log",
                        on_click=clear_history_log,
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                # Scroll area
                ft.Row(
                    controls=[
                        ft.Column(
                            controls=[
                                ft.Row([
                                    # Header columns
                                    ft.Container(content=log_text("ALGO", ft.FontWeight.BOLD), width=50),
                                    ft.Container(content=log_text("MOD", ft.FontWeight.BOLD), width=40),
                                    ft.Container(content=log_text("PREC", ft.FontWeight.BOLD), width=45),
                                    ft.Container(content=log_text("RECL", ft.FontWeight.BOLD), width=45),
                                    ft.Container(content=log_text("MRR", ft.FontWeight.BOLD), width=45),
                                    ft.Container(content=log_text("nDCG", ft.FontWeight.BOLD), width=45),
                                ], spacing=10),
                                ft.Divider(color=ft.Colors.DEEP_PURPLE_200, height=1),
                                history_column,  # Data rows
                            ],
                            scroll=ft.ScrollMode.ADAPTIVE,  # vertical scroll
                            expand=True,
                            alignment=ft.MainAxisAlignment.START,  # keep content on top
                        )
                    ],
                    scroll=ft.ScrollMode.ADAPTIVE,  # horizontal scroll
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START,  # keep column on top
                )
            ],
            spacing=0,
            alignment=ft.MainAxisAlignment.START,  # keep main elements (header, divider, content) on top
        ),
        bgcolor=ft.Colors.DEEP_PURPLE_900,
        padding=15,
        border=ft.Border.all(1, ft.Colors.DEEP_PURPLE_200),
        border_radius=20,
        height=FIXED_CONTAINER_HEIGHT,
        col={"xs": 12, "md": 2.5},
    )

    result_row = ft.ResponsiveRow(
        controls=[
            results_container,
            detail_container,
            history_log_container
        ],
        spacing=25,
        alignment=ft.MainAxisAlignment.START,
        expand=True
    )

    page.add(
        ft.Column(
            controls=[
                title,
                control_grid,
                ft.Column([
                    query_info_display,
                    metrics_display
                ], spacing=5),  # overrule the global setting for less space
                result_row
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=25,  # global setting between container
            expand=True
        )
    )
    handle_dropdown_algo_and_mod(None)

ft.run(main)  # open YAMEx in a separate window
#ft.run(main, view=ft.AppView.WEB_BROWSER) # opens YAMEx in browser