---
title: "Semi-Discrete Optimal Transport for Texture Synthesis"
date: "2024-03-02"
description: Semi-Discrete Optimal Transport in Patch Space for Texture Synthesis
hero: preview.jpg
menu:
    sidebar:
        name: Semi-Discrete OT
        identifier: ot
        weight: 9
tags: ["Texture-Synthesis", "Optimal Transport"]
math: true
---

Texture synthesis is a technique that consists in generating some realistic textures by imitating the patterns of a given exemplar texture. It plays a crucial role in many domains such as computer graphics, where the synthesis quality directly impacts the visual quality and the realism of graphical environments. The objective is to have a realistic replica in a visual sense but with original content: we don’t simply want to copy parts of the original texture for the sake of visual quality and realism.

For the MVA Master, I had the pleasure to work with Paul-Henri Pinart on this subject for the course [Introduction to numerical imaging](https://perso.telecom-paristech.fr/gousseau/MVA/). More specifically, we conducted an in-depth analysis and extensions of the following paper:

> Bruno Galerne, Arthur Leclaire, Julien Rabin. A Texture Synthesis Model Based on Semi-discrete Optimal Transport in Patch Space. SIAM Journal on Imaging Sciences, 2018, ⟨10.1137/18M1175781⟩. ⟨hal-01726443v2⟩

In this paper review, we study the proposed method which uses a patched-based approach. The idea relies on the use of semi-discrete
optimal transport - ie between a continuous and a discrete distribution- to reimpose the patch distribution of the exemplar texture across multiple scales. The goal of this project is to provide quantitative measures of the statistical properties of patches from the synthesis, and understand the method's strengths and weaknesses with qualitative evaluation. We also propose extensions of the method and critically discuss their effectiveness and results compared to the original baseline. Our experiments show that modelling the source distribution with a Gaussian Mixture Model can be simplified by a random sampling of patches in the current synthesis, which reduces the Wasserstein distance between the transported patches and the target distribution. We also propose an alternative method that replaces the semi-discrete optimal transport with affine transport, which greatly simplifies the model and provides equivalent results.

The full paper analysis, our results, extensions and critical perspectives can be accessed in [our report](IIN_report.pdf), with our code publicly available on [(Github)](https://github.com/Clement-W/TextureOptimalTransport_MVA).
