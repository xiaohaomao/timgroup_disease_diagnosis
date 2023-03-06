#
# Primitive makefile
#
BOQABENCHMARK=java -Xmx16G -cp bin:jars/commons-cli-1.2.jar:jars/slf4j-api-1.7.7.jar sonumina.boqa.BOQABenchmark
SIZE_OF_SCORE_DISTRIBUTION=500000

# Defines the directory where to place the results when
# invoking "make install"
DESTDIR ?= /home/sba/workspace/fabn-manuscript/data

ifeq ($(ORPHANET),yes)
ANNOTATIONSFILE=data/phenotype_annotation.omim.orphanet.gz
else
ANNOTATIONSFILE=data/phenotype_annotation.omim.gz
endif

.PHONY: all
all: evaluate

.PHONY: evaluate
evaluate: benchmark evaluate-only

.PHONY: evaluate-only
evaluate-only:
	echo 'source("benchmark-low-noise-s6_load.R");source("src/sonumina/boqa/eval.R")' | R --vanilla
	echo 'source("benchmark-normal-noise-s6_load.R");source("src/sonumina/boqa/eval.R")' | R --vanilla
	echo 'source("benchmark-low-noise-s3_load.R");source("src/sonumina/boqa/eval.R")' | R --vanilla
	echo 'source("benchmark-normal-noise-s3_load.R");source("src/sonumina/boqa/eval.R")' | R --vanilla

.PHONY: benchmark
benchmark: build
	$(BOQABENCHMARK) --alpha 0.001 -r benchmark-low-noise-s6    -o data/human-phenotype-ontology.obo.gz -a $(ANNOTATIONSFILE) -c -m 6 --sizeOfScoreDistribution $(SIZE_OF_SCORE_DISTRIBUTION)
	$(BOQABENCHMARK) --alpha 0.002 -r benchmark-normal-noise-s6 -o data/human-phenotype-ontology.obo.gz -a $(ANNOTATIONSFILE) -c -m 6 --sizeOfScoreDistribution $(SIZE_OF_SCORE_DISTRIBUTION)
	$(BOQABENCHMARK) --alpha 0.001 -r benchmark-low-noise-s3    -o data/human-phenotype-ontology.obo.gz -a $(ANNOTATIONSFILE) -c -m 3 --sizeOfScoreDistribution $(SIZE_OF_SCORE_DISTRIBUTION)
	$(BOQABENCHMARK) --alpha 0.002 -r benchmark-normal-noise-s3 -o data/human-phenotype-ontology.obo.gz -a $(ANNOTATIONSFILE) -c -m 3 --sizeOfScoreDistribution $(SIZE_OF_SCORE_DISTRIBUTION)

.PHONY: build
build:
	ant clean
	ant build

.PHONY: install
install:
	cp *_result.RObj *_param.txt *_summary.txt $(DESTDIR)

.PHONY: dist
dist:
	rm -Rf /tmp/boqa-dist
	mkdir -p /tmp/boqa-dist/boqa /tmp/boqa-dist/boqa/boqa.tests
	cp -Rp . /tmp/boqa-dist/boqa
	cp -Rp boqa.tests /tmp/boqa-dist/
	find /tmp/boqa-dist/  -depth  -type d  -path *.svn* -exec rm -Rf {} ';'
	tar c -C /tmp boqa-dist  | bzip2 >boqa-dist.tar.gz
