import flet as ft

def main(page: ft.Page):
    page.padding = 20
    page.bgcolor = ft.Colors.DEEP_PURPLE_900

    dummy_song_1 = "Dummy song A - Artist 1, 2025"
    dummy_song_2 = "Dummy song B - Artist 2. 1983"
    dummy_song_3 = "Dummy song C - Artist 3, 2023"
    dummy_song_4 = "Dummy song D - Artist 4, 1993"
    dummy_song_5 = "Dummy song E - Artist 3, 2021"
    dummy_song_6 = "Dummy song F - Artist 1, 2003"

    title = ft.Text(
        spans=[
            ft.TextSpan("YAMEx ", ft.TextStyle(size=30, weight=ft.FontWeight.BOLD)),
            ft.TextSpan("- Yet Another Music Explorer", ft.TextStyle(size=20)),
        ],
        text_align=ft.TextAlign.CENTER,
        color=ft.Colors.WHITE,
        expand=True
    )
    title_placement = ft.Row(
        [title],
        alignment=ft.MainAxisAlignment.CENTER,  # horizontal alignment of children
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True
    )

    # result lists of all searches
    results_song = ft.Column(scroll="auto", expand=True)
    results_song.controls.append(ft.Text(dummy_song_1, color=ft.Colors.WHITE))
    results_song.controls.append(ft.Text(dummy_song_2, color=ft.Colors.WHITE))
    results_song.controls.append(ft.Text(dummy_song_3, color=ft.Colors.WHITE))
    results_song.controls.append(ft.Text(dummy_song_4, color=ft.Colors.WHITE))
    results_song.controls.append(ft.Text(dummy_song_5, color=ft.Colors.WHITE))
    results_song.controls.append(ft.Text(dummy_song_6, color=ft.Colors.WHITE))

    results_lyrics = ft.Column(scroll="auto", expand=True)
    results_lyrics.controls.append(ft.Text(dummy_song_2, color=ft.Colors.WHITE))
    results_lyrics.controls.append(ft.Text(dummy_song_4, color=ft.Colors.WHITE))

    results_audio = ft.Column(scroll="auto", expand=True)

    # functions
    def search_song(e):
        results_song.controls.clear()
        results_song.controls.append(ft.Text(f"Top results: {song_input.value}", color=ft.Colors.WHITE))
        page.update()

    def search_lyrics(e):
        results_lyrics.controls.clear()
        results_lyrics.controls.append(ft.Text(f"Top results: {lyrics_input.value}", color=ft.Colors.WHITE))
        page.update()

    def upload_audio(e):
        results_audio.controls.clear()
        results_audio.controls.append(ft.Text("Audio was uploaded!", color=ft.Colors.WHITE))
        page.update()

    # input fields
    song_input = ft.TextField(
        hint_text="Find a song...",
        suffix_icon=ft.Icons.SEARCH_ROUNDED,
        color=ft.Colors.GREY_700,  # text typing color
        bgcolor=ft.Colors.DEEP_PURPLE_50,  # search field color
        border_color=ft.Colors.DEEP_PURPLE_900,  # border color
        border_width=1,
        border_radius=20,
        hover_color=ft.Colors.WHITE,
        focused_bgcolor=ft.Colors.WHITE,
        on_submit=search_song
    )

    lyrics_input = ft.TextField(
        hint_text="Find a song by lyrics...",
        prefix_icon=ft.Icons.SEARCH,
        color=ft.Colors.GREY_700,  # text typing color
        bgcolor=ft.Colors.DEEP_PURPLE_50,  # search field color
        border_color=ft.Colors.DEEP_PURPLE_900,  # border color
        border_width=1,
        border_radius=20,
        hover_color=ft.Colors.WHITE,
        focused_bgcolor=ft.Colors.WHITE,
        on_submit=search_lyrics,
    )

    upload_button = ft.ElevatedButton(
        "Find a song by an audio fragment",
        icon=ft.Icons.UPLOAD,
        on_click=upload_audio,
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.DEEP_PURPLE_50,  # background
            color=ft.Colors.GREY_700,  # text color
            overlay_color=ft.Colors.WHITE,  # hover/pressing color
            side=ft.BorderSide(1, ft.Colors.DEEP_PURPLE_900),  # border
            shape=ft.RoundedRectangleBorder(radius=20),
            padding=20
        ),
    )

    # columns (left, middle, right)
    col_left = ft.Container(
        content=ft.Column(
            [
                song_input,
                results_song,
            ],
            expand=True,
            spacing=10
        ),
        padding=10,
        bgcolor=ft.Colors.DEEP_PURPLE_800,
        border_radius=20,
        expand=True,
    )

    col_middle = ft.Container(
        content=ft.Column(
            [
                lyrics_input,
                results_lyrics,
            ],
            expand=True,
            spacing=10,
        ),
        padding=10,
        bgcolor=ft.Colors.DEEP_PURPLE_800,
        border_radius=20,
        expand=True,
    )

    col_right = ft.Container(
        content=ft.Column(
            [
                upload_button,
                results_audio,
            ],
            expand=True,
            spacing=10
        ),
        padding=10,
        bgcolor=ft.Colors.DEEP_PURPLE_800,
        border_radius=20,
        expand=True,
    )

    # Layout: title + 3 columns
    page.add(
        ft.Column([title_placement]),
        ft.Row(
            [
                col_left,
                col_middle,
                col_right,
            ],
            spacing=20,
            expand=True
        )
    )


ft.app(target=main, view=ft.WEB_BROWSER)
