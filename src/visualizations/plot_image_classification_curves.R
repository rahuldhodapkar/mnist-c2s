#!/usr/bin/env Rscript
#
# Visualize Loss and Accuracy Curves for image_classification experiments
#

library(cowplot)
library(grid)
library(gridExtra)
library(ggpubr)

library(ggplot2)
library(stringr)
library(RColorBrewer)

################################################################################
## Build Output Scaffolding
################################################################################

s=ifelse(!dir.exists("./fig"), dir.create("./fig"), FALSE)
s=ifelse(!dir.exists("./fig/image_classification"), dir.create("./fig/image_classification"), FALSE)

################################################################################
## Load Data
################################################################################

fns <- list.files('./calc/image_classification')
hist.df <- as.data.frame(
    str_match(fns, '^([A-Za-z0-9]+)_([A-Za-z0-9]+)_n([0-9]+)_t([0-9]+).csv$'))

colnames(hist.df) <- c('fn', 'model', 'dataset', 'n_train', 'n_epochs')

hist.df$n_train <- as.numeric(hist.df$n_train)
hist.df$n_epochs <- as.numeric(hist.df$n_epochs)

merged.df <- NULL
for (i in 1:nrow(hist.df)) {
    df <- read.csv(paste0('./calc/image_classification/', hist.df$fn[[i]]))
    df$fn <- hist.df$fn[[i]]
    df$model <- hist.df$model[[i]]
    df$dataset <- hist.df$dataset[[i]]
    df$n_train <- hist.df$n_train[[i]]

    if (is.null(merged.df)) {
        merged.df <- df
    } else {
        merged.df <- rbind(merged.df, df)
    }
}

################################################################################
## Plot Data
################################################################################

datasets <- c('mnist', 'fashionmnist', 'cifar10')
models <- c('distilbert', 'resnet', 'cvt')
n_train <- c(100, 1000, 10000)

get.plot.data <- function(d, m) {
    sub <- subset(merged.df, dataset == d & model == m)
    sub <- subset(sub, type %in% c('loss', 'eval_loss'))
    sub$group <- paste0(sub$type, '; n=',sub$n_train)
    p <- ggplot(
            sub,
            aes(x=epoch, y=value, color=group)) +
        geom_line() + 
        ggtitle(paste0(d," | ", m)) +
        scale_color_manual(
            values=c(brewer.pal('Reds', n=5)[3:5], brewer.pal('Blues', n=5)[3:5]))
    return(p)
}

## 
# Generate loss plots
##

p11 <- get.plot.data('mnist', 'distilbert') +
    ylim(c(0, 3.5)) +
    theme_minimal_grid() + theme(legend.position='none')
p12 <- get.plot.data('mnist', 'resnet') +
    ylim(c(0, 3.5)) +
    theme_minimal_grid() + theme(legend.position='none')
p13 <- get.plot.data('mnist', 'cvt') +
    ylim(c(0, 3.5)) +
    theme_minimal_grid() + theme(legend.position='none')
p1.legend <- as_ggplot(cowplot::get_legend(get.plot.data('mnist', 'cvt')
                       + theme_minimal_grid()))
p1 <- plot_grid(p11, p12, p13, p1.legend,
    labels = c('', '', '', ''),
    rel_widths = c(1, 1, 1, 0.5),
    ncol=4, align='h')


p21 <- get.plot.data('fashionmnist', 'distilbert') +
    ylim(c(0, 3.5)) +
    theme_minimal_grid() + theme(legend.position='none')
p22 <- get.plot.data('fashionmnist', 'resnet') +
    ylim(c(0, 3.5)) +
    theme_minimal_grid() + theme(legend.position='none')
p23 <- get.plot.data('fashionmnist', 'cvt') +
    ylim(c(0, 3.5)) +
    theme_minimal_grid() + theme(legend.position='none')
p2.legend <- as_ggplot(cowplot::get_legend(get.plot.data('fashionmnist', 'cvt')
                       + theme_minimal_grid()))
p2 <- plot_grid(p21, p22, p23, p2.legend,
    labels = c('', '', '', ''),
    rel_widths = c(1, 1, 1, 0.5),
    ncol=4, align='h')



p31 <- get.plot.data('cifar10', 'distilbert') +
    ylim(c(0, 15)) +
    theme_minimal_grid() + theme(legend.position='none')
p32 <- get.plot.data('cifar10', 'resnet') +
    ylim(c(0, 15)) +
    theme_minimal_grid() + theme(legend.position='none')
p33 <- get.plot.data('cifar10', 'cvt') +
    ylim(c(0, 15)) +
    theme_minimal_grid() + theme(legend.position='none')
p3.legend <- as_ggplot(cowplot::get_legend(get.plot.data('cifar10', 'cvt')
                       + theme_minimal_grid()))
p3 <- plot_grid(p31, p32, p33, p3.legend,
    labels = c('', '', '', ''),
    rel_widths = c(1, 1, 1, 0.5),
    ncol=4, align='h')


p <- plot_grid(p1,p2,p3,labels=c('','',''),ncol=1)
ggsave('./fig/image_classification/loss_plots.png', width=13, height=10)

## 
# Generate accuracy plots
##

get.plot.data <- function(d, m) {
    sub <- subset(merged.df, dataset == d & model == m)
    sub <- subset(sub, type %in% c('eval_accuracy'))
    sub$group <- paste0('n=',sub$n_train)
    p <- ggplot(sub, aes(x=epoch, y=value, color=group)) +
        geom_line() + 
        ggtitle(paste0(d," | ", m)) +
        scale_color_manual(
            values=brewer.pal('Oranges', n=5)[3:5])
    return(p)
}

p11 <- get.plot.data('mnist', 'distilbert') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p12 <- get.plot.data('mnist', 'resnet') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p13 <- get.plot.data('mnist', 'cvt') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p1.legend <- as_ggplot(cowplot::get_legend(get.plot.data('mnist', 'cvt')
                       + theme_minimal_grid()))
p1 <- plot_grid(p11, p12, p13, p1.legend,
    labels = c('', '', '', ''),
    rel_widths = c(1, 1, 1, 0.5),
    ncol=4, align='h')
p1


p21 <- get.plot.data('fashionmnist', 'distilbert') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p22 <- get.plot.data('fashionmnist', 'resnet') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p23 <- get.plot.data('fashionmnist', 'cvt') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p2.legend <- as_ggplot(cowplot::get_legend(get.plot.data('fashionmnist', 'cvt')
                       + theme_minimal_grid()))
p2 <- plot_grid(p21, p22, p23, p2.legend,
    labels = c('', '', '', ''),
    rel_widths = c(1, 1, 1, 0.5),
    ncol=4, align='h')
p2


p31 <- get.plot.data('cifar10', 'distilbert') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p32 <- get.plot.data('cifar10', 'resnet') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p33 <- get.plot.data('cifar10', 'cvt') +
    ylim(c(0, 1)) +
    geom_hline(yintercept=0.1, linetype='dashed') +
    theme_minimal_grid() + theme(legend.position='none')
p3.legend <- as_ggplot(cowplot::get_legend(get.plot.data('cifar10', 'cvt')
                       + theme_minimal_grid()))
p3 <- plot_grid(p31, p32, p33, p3.legend,
    labels = c('', '', '', ''),
    rel_widths = c(1, 1, 1, 0.5),
    ncol=4, align='h')
p3

p <- plot_grid(p1,p2,p3,labels=c('','',''),ncol=1)
p
ggsave('./fig/image_classification/accuracy_plots.png', width=13, height=10)

print('All done!')
