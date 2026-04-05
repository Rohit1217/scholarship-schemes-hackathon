#!/bin/bash
# Download NSP scheme guideline PDFs with polite delays
# Usage: bash scripts/download_scheme_pdfs.sh

OUTDIR="data/pdfs"
mkdir -p "$OUTDIR"

BASE="https://scholarships.gov.in/public/schemeGuidelines"

# Only Specifications (guideline) PDFs — skip FAQs, user manuals, nodal officers
PDFS=(
  "NMMSSGuidelines.pdf"
  "CSSS_GUIDLINES_07022024_updated.pdf"
  "DEPDGuidelines_1.pdf"
  "Top_Class_Education_Scheme_2018.pdf"
  "ApprovedmodifieddraftofTopClassCollege.pdf"
  "tribalfellowshipguideline.pdf"
  "Guidelines_ISHAN UDAY_2324.pdf"
  "Guidelines_NATIONAL_SCHOLARSHIP_FOR_POSTGRADUATE_STUDIES_UGC_2324.pdf"
  "GuidelinesforICAR_NTS_Scholarship_3103_3104.pdf"
  "RevisedGuidelinesforICAR_JRF_SRFandPG_Scholarship_3105_3106.pdf"
  "NEC_1234_G.pdf"
  "STIPEND_STATICAL_GUIDLLINES.pdf"
  "MORB_GL.pdf"
  "3061_G.pdf"
  "Labour_Ministry_Guidelines_of_scholarhsips schemes.pdf"
  "warb/PMSS_Guidelines_1197_3001-2023-24.pdf"
  "AICTE/AICTE_2010_G.pdf"
  "AICTE/AICTE_2011_G.pdf"
  "AICTE/AICTE_2012_G.pdf"
  "AICTE/AICTE_2013_G.pdf"
  "AICTE/AICTE_3038_G.pdf"
  "AICTE/AICTE_3039_G.pdf"
  "AICTE/PM_USPY(SSSJKL)SchemeSpecifications.pdf"
)

echo "Downloading ${#PDFS[@]} scheme guideline PDFs to $OUTDIR/"
echo ""

for PDF in "${PDFS[@]}"; do
  FILENAME=$(basename "$PDF")
  OUTFILE="$OUTDIR/$FILENAME"
  URL="$BASE/$PDF"

  if [ -f "$OUTFILE" ]; then
    echo "  [skip] $FILENAME (already exists)"
    continue
  fi

  echo "  Downloading: $FILENAME"
  wget -q --user-agent="Mozilla/5.0 Chrome/120" \
       --timeout=30 --tries=2 \
       -O "$OUTFILE" "$URL"

  if [ $? -eq 0 ] && [ -s "$OUTFILE" ]; then
    SIZE=$(du -h "$OUTFILE" | cut -f1)
    echo "    OK ($SIZE)"
  else
    echo "    FAILED — removing"
    rm -f "$OUTFILE"
  fi

  sleep 2  # polite delay
done

echo ""
echo "Done. PDFs in $OUTDIR/:"
ls -lh "$OUTDIR/"
