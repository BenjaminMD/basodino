# latexmk: LuaLaTeX + biber; auxiliary files in build/, main.pdf next to main.tex.
$pdf_mode = 4;                 # 4 = lualatex
$lualatex = 'lualatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$bibtex_use = 2;               # run biber when needed; -c also removes the .bbl
$aux_dir = 'build';
$out_dir = '.';
@default_files = ('main.tex');
