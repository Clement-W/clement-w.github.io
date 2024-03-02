---
title: "On the Curvature of Deep Generative Models "
date: "2024-03-02"
description: Use Riemannian Geometry to understand and manipulate the latent space of VAEs
hero: preview.png
menu:
    sidebar:
        name: Latent Space Oddity
        identifier: latentspaceoddity
        weight: 10
tags: ["Generative Models", "Riemannian Geometry"]
math: true
---

According to the manifold hypothesis, high-dimensional data, despite its apparent complexity, often adheres to a simpler underlying structure, typically represented as a manifold, or a lower-dimensional surface embedded within the larger dimensional space.
Performing computations within these high-dimensional environments presents significant challenges. A practical approach is to parameterize the surface in $\mathcal{X}$ by a low-dimensional variable $\mathbf{z} \in \mathcal{Z}$ created with a suitable smooth generator function $f : \mathcal{Z} \to \mathcal{X}$. This generator defines a surface in the input space from a latent representation which remains difficult to interpret in terms of geometry. For deep generative models, this approach opens the door to a new way to analyze and interpret the data representation.

For the MVA Master, I had the pleasure to work with Théo Danielou and Elie Bakouch on this subject for the course [Geometric Data Analysis](https://www.jeanfeydy.com/Teaching/index.html). More specifically, we conducted an in-depth analysis and extensions of the following paper:

> Arvanitidis, G., Hansen, L. K., & Hauberg, S. (2017). Latent space oddity: on the curvature of deep generative models. arXiv preprint arXiv:1710.11379.

We explored the author’s approach of using Riemannian geometry to understand and manipulate the latent spaces of Variational Autoencoders (VAEs). The non-linearity of the generator (or decoder) of these models results in a latent space that presents a distorted representation of the input space. The study addresses this distortion by characterizing it with an approximation of a Riemannian metric that benefits from an original generator architecture designed to enhance the accuracy of variance estimates.

[Our report](GDA_report.pdf) examines the theoretical foundations and methodologies proposed for estimating the curvature in the latent space of these models. We further deepen our analysis through experiments showcasing the efficacy and relevance of the method, concluding with a discussion on its potential limitations. Our code is publicly available on [(Github)](https://github.com/Clement-W/latent_space_oddity_MVA).
