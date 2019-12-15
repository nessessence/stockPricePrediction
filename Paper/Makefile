NAME = conference

all:
	latex ${NAME}.tex
	bibtex ${NAME}
	latex ${NAME}
	latex ${NAME}

ps: all
	dvips -M -t letter -Ppdf -G0 ${NAME}.dvi -o ${NAME}.ps

pdf: ps
	ps2pdf -dPDFSETTINGS=/printer -dCompatibilityLevel=1.3 -dMaxSubsetPct=100 -dSubsetFonts=true -dEmbedAllFonts=true -sPAPERSIZE=a4 ${NAME}.ps

view: all
	xdvi ${NAME}.dvi

# Make and open the PDF in OS X.
viewpdf: pdf
	open ${NAME}.pdf

clean:
	rm -rf *.bbl *.blg *.aux *.log *~ *.bak ${NAME}.ps ${NAME}.dvi ${NAME}.log ${NAME}.pdf figure/*.bak beamish.ece.cmu.edu \#* 

spell:
	ispell -f ispell.words -t *.tex