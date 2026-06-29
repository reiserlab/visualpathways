# Visual Pathways Analysis

This repository contains analysis code and companion resources for studying visual pathways in the *Drosophila* Male CNS connectome ([Janelia FlyEM](https://neuprint-cns.janelia.org/?dataset=cns&qt=findneurons)).

The corresponding preprint is [The organization of visual pathways in the *Drosophila* brain](https://www.biorxiv.org/content/10.64898/2025.12.22.696097v2.abstract).

## Interactive Resources

- Interactive HTML figures are available through the [GitHub Pages figure browser](https://reiserlab.github.io/visualpathways/results/html_figures/).

- [Visual Central Brain Neuron gallery](https://reiserlab.github.io/visualpathways/gallery_vcbn/): a browsable cell-type-level gallery for predicted visual central brain neurons (VCBNs). Each page summarizes cell count, predicted neurotransmitter, VCBN pathway class, median layer, median Visual Input Contribution (VIC), median alpha-hull size, visual-input barcode, pathway summary, anterior-view morphology, and predicted anatomical receptive field (ARF). Use this resource to inspect individual VCBN predictions and generate hypotheses about upstream visual inputs and spatial sampling.
- [Interactive Fig. 6c Neuroglancer view](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/auxiliary-data/retinotopy-tbars/retinotopy-tbars.json): an interactive companion to the Fig. 6c visualization. The view shows retinotopy-annotated presynaptic sites (T-bars), enabling inspection of the synaptic locations and retinotopic coordinates used to map spatial visual information across the central brain. Use the Neuroglancer layer, `ME(R)-columns-r-theta`, as a color reference.
