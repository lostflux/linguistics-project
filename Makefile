# Minimal Makefile to clean up directories.
#
# Author: Amittai Siavava
# Date: 2022-09-23

clean:
	# delete compiled pdfs
	# @find . -name *.pdf -exec rm -f {} +

	# delete latexmk files
	@find . -name *.fls -exec rm -f {} +
	@find . -name *.fdb_latexmk -exec rm -f {} +

	# delete aux files
	@find . -name *.aux -exec rm -f {} +
	@find . -name *.log -exec rm -f {} +
	@find . -name *.out -exec rm -f {} +
	@find . -name *.toc -exec rm -f {} +

	# delete bbl files
	@find . -name *.bbl -exec rm -f {} +
	@find . -name *.blg -exec rm -f {} +

	# delete notes files
	@find . -name *.nav -exec rm -f {} +
	@find . -name *.snm -exec rm -f {} +

	# delete synctex files
	@find . -name *.synctex.gz -exec rm -f {} +


	# delete backup files
	@find . -name *.backup -exec rm -f {} +
	@find . -name *.backup~ -exec rm -f {} +
	@find . -name *.backup\~ -exec rm -f {} +
	@find . -name *.bak -exec rm -f {} +
	@find . -name *.backup\# -exec rm -f {} +

	# delete other files
	@find . -name *.vrb -exec rm -f {} +
	@find . -name *.lof -exec rm -f {} +
	@find . -name *.lot -exec rm -f {} +
	@find . -name *.gz -exec rm -f {} +
	@find . -name *.dvi -exec rm -f {} +
	@find . -name *.ps -exec rm -f {} +

	# delete .run.xml files
	@find . -name *.run.xml -exec rm -f {} +

	# delete -blx.bib files
	@find . -name *-blx.bib -exec rm -f {} +

.PHONY: clean
