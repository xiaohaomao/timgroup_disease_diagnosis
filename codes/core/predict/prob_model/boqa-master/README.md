# BOQA #

## Introduction ##

This is the code that accompanies the publication "Bayesian
Ontology Querying for Accurate and Noise-Tolerant Semantic
Searches".

## Patent ##

The authors of this software have filed a patent on ontology
search algorithms for several applications, including the
algorithm implemented in this work.

We release the source code under the ClearBSD license. Additional,
we grant the usage of the patent for academic purposes that have no
commercial interests. Other parties have to contact one of the
authors for obtaining permission or a license.

## Benchmark ##

In order to start the benchmark, just invoke the supplied
makefile. You need Java, ant, and R for successfully
completing the benchmark process. Look into the code for
more details about the implementation and how to use it in
your own research project. Remember that it also often helps
to just read the supplied tests to understand how to use an 
API. In particular, the BOQA implementation is tested in
BOQATest class.

## Usage ##

Given a directory ``hpo_dir`` with HPO files (``*_hpo.txt``) and an ouput directory ``out_dir``, and from the root of the repo:
```bash
java -Xmx16G -cp bin:jars/commons-cli-1.2.jar sonumina.boqa.BOQABenchmark -o data/hp.obo.gz -a data/new_phenotype.gz -p <hpo_dir> -d <out_dir>
```

## History ##
Code forked from public release: http://compbio.charite.de/boqa/

Berlin, June 8th, 2012
Sebastian Bauer, Sebastian Köhler, Marcel Schulz, Peter Robinson
