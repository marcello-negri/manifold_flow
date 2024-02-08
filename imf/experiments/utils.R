library("gnorm")

sample_gen_norm <- function(x, alpha, beta, mu=0){
    samples <- rgnorm(x, mu=0, alpha=alpha, beta=beta)
    return (samples)
}