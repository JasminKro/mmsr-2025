library(dplyr)
library(ggplot2)

# Load data
df <- metrics_at10

# Apply specific order to the data frame
df_ordered <- df
df_ordered$algorithm <- factor(df_ordered$algorithm, levels = c(
  "Random",
  "Unimodal_Audio", "Unimodal_Lyrics", "Unimodal_Video",
  "Early_All", "Early_Audio_Lyrics", "Early_Audio_Video", "Early_Lyrics_Video",
  "Late_All", "Late_Audio_Lyrics", "Late_Audio_Video", "Late_Lyrics_Video",
  "nn_audio_audio", "nn_audio_lyrics", "nn_audio_video",
  "nn_lyrics_audio", "nn_lyrics_lyrics", "nn_lyrics_video",
  "nn_video_audio", "nn_video_lyrics", "nn_video_video"))

compare_algorithms <- function(data, algorithms_to_compare, title = "Algorithm Comparison", alg_colors, alg_names) {
  
  # Selected algorithms only
  df_filtered <- data %>% filter(algorithm %in% algorithms_to_compare) %>% 
    filter(metric != "Pop") # do not show popularity
  
  # Mean ± sd for each metric
  p <- ggplot(df_filtered, aes(x = algorithm, y = mean, fill = algorithm)) +
    geom_bar(stat = "identity") + # main bars (identity -> no calculation needed, values provided)
    geom_errorbar( # whiskers for sd
      aes(
        ymin = pmax(0, mean - sd), # use 0 instead of negative value
        ymax = pmin(1, mean + sd)  # use 1 if value > 1
      ),
      width = 0.2, # width of horizontal lines
    ) +
    facet_wrap(~ metric, scales = "free_y") + # separate graphs, split by metric, use y values best suited to the metric
    theme_minimal() +
    theme(
      axis.text.x = element_blank(), # no labels under the bars
      axis.ticks.x = element_blank(), # no lines sticking out at the bottom
      strip.text = element_text(size = 15, face = "bold"), # facet labels
      legend.position = "bottom",
      legend.text = element_text(size = 15), # algorithm names
      text = element_text(size = 14)
    ) +
    labs(title = title, x = "", y = "") + # no axis labels
    guides(fill=guide_legend(nrow=2,byrow=FALSE)) + # show legend in two rows
    scale_fill_manual(name = NULL, # no plot title
                      values = alg_colors, labels = alg_names) # colors and renaming for plot
  
  print(p)
}

# Main algorithms
compare_algorithms( 
  df_ordered,
  algorithms_to_compare = c("Random", "Unimodal_Audio", "Early_Audio_Lyrics", "Late_All", "nn_video_audio"),
  title = "", 
  c(
    "nn_video_audio" = "red3", 
    "Late_All" = "steelblue2",
    "Early_Audio_Lyrics" = "darkorange2", #"Early_All" = "darkorange2",
    "Unimodal_Audio" = "limegreen", 
    "Random" = "grey40"
  ),
  c(
    "nn_video_audio" = "Neural Network (Video - Audio)",
    "Late_All" = "Late Fusion (Audio + Lyrics + Video)",
    "Early_Audio_Lyrics" = "Early Fusion (Audio + Lyrics)", # "Early_All" = "Early Fusion (Audio + Lyrics + Video)",
    #"Early_All" = "Audio + Lyrics + Video",
    "Unimodal_Audio" = "Unimodal (Audio)",
    "Random" = "Random Baseline"
  )
)

# Unimodal algorithms
compare_algorithms(
  df_ordered,
  algorithms_to_compare = c("Unimodal_Audio", "Unimodal_Lyrics", "Unimodal_Video"),
  title = "",
  c("Unimodal_Audio" = "limegreen", "Unimodal_Lyrics" = "darkseagreen2", "Unimodal_Video" = "darkseagreen"),
  c("Unimodal_Audio" = "Audio", "Unimodal_Lyrics" = "Lyrics", "Unimodal_Video" = "Video")
)

# Early fusion
compare_algorithms(
  df_ordered,
  algorithms_to_compare = c("Early_All", "Early_Audio_Lyrics", "Early_Audio_Video", "Early_Lyrics_Video"),
  title = "",
  c("Early_All" = "peachpuff", "Early_Audio_Lyrics" = "darkorange2", "Early_Audio_Video" = "peachpuff3", "Early_Lyrics_Video" = "peachpuff4"),
  c("Early_All" = "Audio + Lyrics + Video", "Early_Audio_Lyrics" = "Audio + Lyrics", "Early_Audio_Video" = "Audio + Video", "Early_Lyrics_Video" = "Lyrics + Video")
)

# Late fusion
compare_algorithms(
  df_ordered,
  algorithms_to_compare = c("Late_All", "Late_Audio_Lyrics", "Late_Audio_Video", "Late_Lyrics_Video"),
  #title = "Late Fusion Modality Comparison"
  title = "",
  c("Late_All" = "steelblue2", "Late_Audio_Video" = "lightblue2", "Late_Audio_Lyrics" = "lightblue3", "Late_Lyrics_Video" = "lightblue4"),
  c("Late_All" = "Audio + Lyrics + Video", "Late_Audio_Video" = "Audio + Video", "Late_Audio_Lyrics" = "Audio + Lyrics", "Late_Lyrics_Video" = "Lyrics + Video")
)

# Neural network
compare_algorithms_nn <- function(data, algorithms_to_compare, title = "Algorithm Comparison") {
  
  # Selected algorithms only
  df_filtered <- data %>% filter(algorithm %in% algorithms_to_compare) %>% 
    filter(metric != "Pop")
  
  # Mean ± sd for each metric
  p <- ggplot(df_filtered, aes(x = algorithm, y = mean, fill = algorithm)) +
    geom_bar(stat = "identity") + # main bars (identity -> no calculation needed, values provided)
    geom_errorbar( # whiskers for sd
      aes(
        ymin = pmax(0, mean - sd), # use 0 instead of negative value
        ymax = pmin(1, mean + sd)  # use 1 if value > 1
      ),
      width = 0.2, # width of horizontal lines
    ) +
    facet_wrap(~ metric, scales = "free_y") + # separate graphs, split by metric, use y values best suited to the metric
    theme_minimal() +
    theme(
      axis.text.x = element_blank(), # no labels under the bars
      axis.ticks.x = element_blank(), # no lines sticking out at the bottom
      strip.text = element_text(size = 15, face = "bold"), # facet labels
      legend.position = "bottom",
      legend.text = element_text(size = 15), # algorithm names
      text = element_text(size = 14)
    ) +
    labs(fill = NULL, title = title, x = "", y = "") # no axis labels
  
  print(p)
}

compare_algorithms_nn (
  df_ordered,
  algorithms_to_compare = c(
    "nn_audio_audio", "nn_audio_lyrics", "nn_audio_video",
    "nn_lyrics_audio", "nn_lyrics_lyrics", "nn_lyrics_video",
    "nn_video_audio", "nn_video_lyrics", "nn_video_video"
  ),
  title = ""
)
