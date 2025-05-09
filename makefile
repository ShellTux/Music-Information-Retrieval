ARCHIVE        = Multimedia-PL2-G1.zip
REPORT         = docs/relatorio.pdf

PANDOC_OPTS += --resource-path=docs
PANDOC_OPTS += --filter=pandoc-include
PANDOC_OPTS += --citeproc

%.pdf: %.typ
	# pandoc $(PANDOC_OPTS) --output=$@ $<
	typst compile $<

$(ARCHIVE): $(REPORT) $(IMAGES)
	rm --force $@
	git ls-files > files.txt
	ls -1 $^ >> files.txt
	sort --output=files.txt files.txt
	zip -@ $@ < files.txt

.PHONY: archive
archive: $(ARCHIVE)
