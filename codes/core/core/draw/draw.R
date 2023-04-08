library(ggplot2)
library(rjson)


jsonToDataFrame <- function(fileName){
    lst <- fromJSON(file=fileName)
    return(as.data.frame(lst))
}

draw_quartile <- function(para_list){
    pl <- para_list
    df <- data.frame()
    for (json_path in pl[['jsonPaths']])
        df <- rbind(df, jsonToDataFrame(json_path))
    fig <- ggplot(df, aes_string(x=pl[['xCol']], y=pl[['yCol']], fill=pl[['classCol']]))
    fig <- fig + geom_boxplot(outlier.size=1.5, outlier.shape=21)
    yLim <- boxplot.stats(df[, pl[['yCol']]])$stats[c(1, 5)]
    fig <- fig + coord_cartesian(ylim = yLim*4)
    cat('yLim:', yLim)

    ggsave(pl[['figPath']], fig, width=40, height=20, units="cm")
}

draw_multi_line_from_df <- function(para_list){
    pl <- para_list
    df <- read.csv(pl[['csvPath']])
    fig <- ggplot(df, aes_string(x=pl[["xCol"]], y=pl[["yCol"]], colour=pl[['classCol']])) + geom_line()
    if (!is.null(pl[["classOrder"]]))
        fig <- fig + scale_colour_discrete(limits=pl[['classOrder']])   #
    ggsave(pl[['figPath']], fig, width = 40, height = 20, units = "cm")
}

draw_multi_line <- function(para_list){
    pl <- para_list
    df <- data.frame()
    for (json_path in pl[['jsonPaths']])
        df <- rbind(df, jsonToDataFrame(json_path))
#    print(df)
    fig <- ggplot(df, aes_string(x=pl[["xCol"]], y=pl[["yCol"]], colour=pl[['classCol']], group=pl[['classCol']])) + geom_line()
    if (!is.null(pl[["classOrder"]]))
        fig <- fig + scale_colour_discrete(limits=pl[['classOrder']])   #
    ggsave(pl[['figPath']], fig, width = 40, height = 20, units = "cm")
}

draw_dodge_bar <- function(para_list){
    pl <- para_list
    df <- data.frame()
    for (json_path in pl[['jsonPaths']])
        df <- rbind(df, jsonToDataFrame(json_path))
    fig <- ggplot(df, aes_string(x=pl[["xCol"]], y=pl[["yCol"]], fill=pl[['classCol']])) + geom_bar(position="dodge", stat="identity")
    fig <- fig + scale_fill_discrete(limits=pl[['classOrder']])
    if (!is.null(pl[["classOrder"]]))
        fig <- fig + scale_colour_discrete(limits=pl[['classOrder']])   #
    if (!is.null(pl[['xOrder']]))
        fig <- fig + scale_x_discrete(limits=pl[['xOrder']])
    ggsave(pl[['figPath']], fig, width = 40, height = 20, units = "cm")
}


args <- commandArgs()
para_list <- fromJSON(file=args[6])
if (para_list[['drawFunc']] == 'draw_quartile')
    draw_quartile(para_list)
if (para_list[['drawFunc']] == 'draw_multi_line')
    draw_multi_line(para_list)
if (para_list[['drawFunc']] == 'draw_multi_line_from_df')
    draw_multi_line_from_df(para_list)
if (para_list[['drawFunc']] == 'draw_dodge_bar')
    draw_dodge_bar(para_list)
