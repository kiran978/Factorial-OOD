from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path
import pandas as pd, json

BASE=Path('/mnt/data/phase5_final')
out=BASE/'Factorial_OOD_Phase5_Leakage_Safe_Feature_Report_v1.0.docx'
summary=json.load(open(BASE/'outputs/P5_summary.json'))
pca=pd.read_csv(BASE/'outputs/P5_07_HTTr_PCA_scope_summary.csv')
gate=pd.read_csv(BASE/'outputs/P5_12_phase5_completion_decision.csv')
ind=pd.read_csv(BASE/'validation/P5_11_independent_feature_and_leakage_validation.csv')
local=pd.read_csv(BASE/'validation/P5_10_local_Phase5B_leakage_tests.csv')

doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(0.65); sec.bottom_margin=Inches(0.65); sec.left_margin=Inches(0.75); sec.right_margin=Inches(0.75)
styles=doc.styles
styles['Normal'].font.name='Arial'; styles['Normal'].font.size=Pt(9.5)
for st in ['Title','Heading 1','Heading 2']:
    styles[st].font.name='Arial'
styles['Title'].font.size=Pt(20); styles['Title'].font.bold=True
styles['Heading 1'].font.size=Pt(14); styles['Heading 1'].font.bold=True
styles['Heading 2'].font.size=Pt(11); styles['Heading 2'].font.bold=True

def shade(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_cell_text(cell, text, bold=False, size=8.5):
    cell.text=''; p=cell.paragraphs[0]; r=p.add_run(str(text)); r.bold=bold; r.font.name='Arial'; r.font.size=Pt(size); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(headers, rows, widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for i,h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i],h,True,8); shade(t.rows[0].cells[i],'D9EAF7')
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row): set_cell_text(cells[i],v,False,8)
    return t

p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run('FACTORIAL-OOD\nPhase 5 — Leakage-Safe Feature Construction')
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Final closure report v1.0 | Protocol v0.2 | 31 August 2026'); r.bold=True; r.font.size=Pt(10)

p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('PHASE 5 STATUS: PASS — PHASE 6 MODELING AUTHORIZED'); r.bold=True; r.font.size=Pt(12)

# Executive summary
h=doc.add_heading('1. Executive decision',level=1)
p=doc.add_paragraph()
p.add_run('Phase 5 is formally closed. ').bold=True
p.add_run('The structure-primary, HTTr-primary, and fusion-ready feature infrastructure was completed under the frozen outer/inner scopes from Gate C. No Tox21 endpoint values or predictive-performance results were used to learn any feature transformation.')

rows=[
('Curated parents','2,017'),('Route-B parent–context rows','4,021'),('Structure representation','2,048 Morgan bits + 10 physicochemical descriptors'),('HTTr raw variables','16,821'),('HTTr PCA scopes','5 outer + 20 inner = 25'),('Retained HTTr PCs','256 in every scope'),('Early-fusion dimension','2,314'),('Local leakage/QA tests','20/20 PASS'),('Independent revalidation','20/20 PASS')]
add_table(['Item','Frozen result'],rows)

# Structure
h=doc.add_heading('2. Structure-primary representation',level=1)
p=doc.add_paragraph('All 2,017 curated parent structures parsed successfully with RDKit. Morgan fingerprints are identity-derived and therefore fixed globally; no outcome data enter fingerprint generation. The compact descriptor panel is raw at source and standardized only within the corresponding training scope using unique training parents.')
add_table(['Component','Implementation'],[
('Morgan fingerprint','Radius 2; 2,048 bits; chirality disabled; unscaled binary representation'),
('Physicochemical descriptors','MW, LogP, TPSA, HBD, HBA, rotatable bonds, ring count, fraction Csp3, heavy-atom count, formal charge'),
('Descriptor scaling','25 scope-specific StandardScaler fits: 5 outer final-refit + 20 inner training scopes'),
('Raw structure dimension','2,058 features')])

# HTTr
h=doc.add_heading('3. HTTr training-scope PCA',level=1)
p=doc.add_paragraph('The uploaded Phase 5B computation used only metadata plus HTTr values from the checksum-locked Feather matrix. Tox21 columns were counted for schema QA but their values were not loaded into the PCA path. Duplicate parent–context source rows were collapsed featurewise by the frozen median rule before transformation.')

outer=pca[pca['inner_fold'].astype(str)=='FINAL_REFIT'].copy().sort_values('outer_fold')
rows=[]
for _,r in outer.iterrows():
    rows.append((int(r.outer_fold),int(r.fit_parent_n),int(r.fit_context_rows),int(r.retained_httr_features),int(r.zero_variance_httr_features),int(r.pca_components_retained),f"{100*r.cumulative_variance_at_max_components:.2f}%"))
add_table(['Outer fold','Train parents','Train context rows','HTTr retained','Zero-var','PCs','Variance @256'],rows)
p=doc.add_paragraph(f"Across all 25 scopes, cumulative variance at the 256-PC cap ranged from {100*summary['pca_cumulative_variance_range'][0]:.2f}% to {100*summary['pca_cumulative_variance_range'][1]:.2f}%. Thus the 80% target was not reached before the prespecified 256-component cap, so all scopes retain 256 PCs.")

# Fusion
h=doc.add_heading('4. Fusion-ready infrastructure',level=1)
add_table(['Feature family','Frozen implementation'],[
('Structure-only','Morgan 2,048 + 10 scope-standardized descriptors'),
('HTTr-only','256 fold-specific PCA scores'),
('Early fusion','[Morgan 2,048 | descriptors 10 | HTTr PCs 256] = 2,314 features'),
('Late fusion','Separate structure and HTTr probabilities; convex weight selected within inner grouped CV only'),
('Route-B aggregation','Mean context-row probability → parent probability (primary); median as sensitivity'),
('Uncertainty unit','Compound/parent-level resampling and inference')])

# QA
h=doc.add_heading('5. Leakage and integrity validation',level=1)
p=doc.add_paragraph('The local run reported 20/20 PASS. The uploaded archive was then independently rechecked rather than accepted from its local status flags alone.')
important=ind[['test_id','description','status','detail']]
rows=[tuple(x) for x in important.values.tolist()]
add_table(['Test','Independent check','Status','Detail'],rows)

# Hash and reproducibility
h=doc.add_heading('6. Reproducibility lock',level=1)
p=doc.add_paragraph('Uploaded Phase5B package SHA-256:')
p.style=doc.styles['Normal']; r=p.add_run('699dc521e0be0906fbc9b270ba08106874a0aeeaf96b2936453d473af4eb8894'); r.font.name='Courier New'; r.font.size=Pt(8.5)
p=doc.add_paragraph('The 25 full PCA transformer NPZ files remain on the local analysis workstation under Phase5B_Output/transformers. Their immutable SHA-256 hashes are carried in the Phase 5 transformer registry. The closure package intentionally does not duplicate approximately 0.406 GiB of transformer arrays.')

# Completion gate
h=doc.add_heading('7. Phase 5 completion decision',level=1)
add_table(['Component','Status','Evidence'],[tuple(x) for x in gate.values.tolist()])
p=doc.add_paragraph()
p.add_run('Decision: PASS. ').bold=True
p.add_run('Phase 6 — Locked Baseline and Multimodal Model Benchmark may begin using the frozen primary outer folds, nested inner folds, feature specifications, transformer checksums, and fusion rules. No feature transformation may be refitted using outer-test rows or endpoint performance.')

# Footer
for section in doc.sections:
    fp=section.footer.paragraphs[0]; fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    rr=fp.add_run('FACTORIAL-OOD | Phase 5 closure | Protocol v0.2'); rr.font.name='Arial'; rr.font.size=Pt(8)

doc.save(out)
print(out)
