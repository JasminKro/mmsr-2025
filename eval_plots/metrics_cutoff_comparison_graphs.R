library(tidyverse)
library(dplyr)
library(ggplot2)

# Load data
df <- retrieval_metrics

# Apply specific order to the data frame
df_ordered <- df
df_ordered$algorithm <- factor(df_ordered$algorithm, levels = c("Random Baseline", "Unimodal (Audio)", "Early Fusion", "Late Fusion", "Neural Network"))

# Filter out the popularity because it behaves differently
df_ordered %>% 
  filter(metric != "Pop") %>% 
  ggplot(aes(x = k, y = mean, color = algorithm)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.5) +
    facet_wrap(~ metric, scales = "free_y") + # separate graphs, split by metric, use y values best suited to the metric
    scale_x_continuous( # else low k values too close together
      trans = "log10", # distance based on ratio instead of linear
      breaks = c(5, 10, 20, 50, 100, 200),
      labels = c("5", "10", "20", "50", "100", "200")
    ) +
    labs(
      title = "",
      x = "k", 
      y = "Metric value (mean)"
    ) +
    theme_minimal() + # alternative for slides: theme_light()
    theme(
      text = element_text(size = 14), # base font size
      axis.title = element_text(size = 16), # 'k' and 'Metric value'
      axis.text = element_text(size = 12), # axis numbers
      strip.text = element_text(size = 14, face = "bold"), # facet labels
      legend.text = element_text(size = 14), # algorithm names
      legend.title = element_text(size = 15, face = "bold") # 'algorithm'
    ) + 
    scale_color_manual(values = c("Neural Network" = "red3", 
                                  "Late Fusion" = "steelblue2", 
                                  "Early Fusion" = "darkorange2", 
                                  "Unimodal (Audio)" = "limegreen", 
                                  "Random Baseline" = "grey40"))

# Popularity
df_ordered %>% 
  filter(metric == "Pop") %>% 
  ggplot(aes(x = k, y = mean, color = algorithm)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_x_continuous( # else low k values too close together
    trans = "log10", # distance based on ratio instead of linear
    breaks = c(5, 10, 20, 50, 100, 200),
    labels = c("5", "10", "20", "50", "100", "200")
  ) +
  labs(
    title = "",
    x = "k", 
    y = "Popularity value (mean)"
  ) +
  theme_light() + # alternative for report: theme_minimal()
  theme(
    text = element_text(size = 14), # base font size
    axis.title = element_text(size = 16), # 'k' and 'Metric value'
    axis.text = element_text(size = 12), # axis numbers
    legend.text = element_text(size = 14), # algorithm names
    legend.title = element_text(size = 15, face = "bold") # 'algorithm'
  ) + 
  scale_color_manual(values = c("Neural Network" = "red3", 
                                "Late Fusion" = "steelblue2", 
                                "Early Fusion" = "darkorange2", 
                                "Unimodal (Audio)" = "limegreen", 
                                "Random Baseline" = "grey40"))
