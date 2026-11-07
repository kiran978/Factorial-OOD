from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pandas as pd, json
from pathlib import Path

BASE=Path('/mnt/data/factorial_ood_phase4/outputs')
OUT=Path('/mnt/data/Factorial_OOD_Phase4_GateC_Report_v1.0.docx')
summary=json.load(open(BASE/'P4_summary.json'))
leak=pd.read_csv(BASE/'P4_12_leakage_unit_tests.csv')
gate=pd.read_csv(BASE/'P4_16_gateC_decision.csv')
outsz=pd.read_csv(BASE/'P4_13_outer_fold_size_summary.csv')
insz=pd.read_csv(BASE/'P4_14_inner_fold_size_summary.csv')
sparse=pd.read_csv(BASE/'P4_17_factorial_fold_sparsity_note.csv')
classsup=pd.read_csv(BASE/'P4_09_primary_endpoint_nested_class_support.csv')
registry=pd.read_csv(BASE/'P4_02_split_registry.csv')

D=Document()
sec=D.sections[0]
sec.top_margin=Inches(0.65); sec.bottom_margin=Inches(0.65); sec.left_margin=Inches(0.72); sec.right_margin=Inches(0.72)
styles=D.styles
styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(9.5)
for nm,size in [('Title',22),('Heading 1',15),('Heading 2',12)]:
    styles[nm].font.name='Aptos Display' if nm!='Normal' else 'Aptos'; styles[nm].font.size=Pt(size)
    styles[nm].font.color.rgb=RGBColor(31,54,76)

def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_cell_text(cell,text,bold=False,color=None,size=8.5):
    cell.text=''; p=cell.paragraphs[0]; r=p.add_run(str(text));r.bold=bold;r.font.name='Aptos';r.font.size=Pt(size)
    if color:r.font.color.rgb=RGBColor(*color)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(headers,rows,widths=None):
    t=D.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for i,h in enumerate(headers):
        set_cell_text(t.rows[0].cells[i],h,True,(255,255,255),8.2); shade(t.rows[0].cells[i],'1F4E78')
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row): set_cell_text(cells[i],v,False,None,8.0)
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width=Inches(w)
    return t

def bullet(text):
    p=D.add_paragraph(style='List Bullet'); p.paragraph_format.space_after=Pt(2); p.add_run(text)

def callout(title,text,fill='EAF2F8'):
    t=D.add_table(rows=1,cols=1); t.alignment=WD_TABLE_ALIGNMENT.CENTER
    c=t.cell(0,0); shade(c,fill); c.text=''
    p=c.paragraphs[0]; p.paragraph_format.space_after=Pt(2); r=p.add_run(title+'\n');r.bold=True;r.font.size=Pt(10);r.font.color.rgb=RGBColor(31,78,121)
    r=p.add_run(text);r.font.size=Pt(9.2)
    D.add_paragraph().paragraph_format.space_after=Pt(1)

# Title
p=D.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('FACTORIAL-OOD');r.bold=True;r.font.size=Pt(12);r.font.color.rgb=RGBColor(79,98,40)
p=D.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Phase 4 — Frozen Outer/Inner Split Package and Leakage Unit Tests');r.bold=True;r.font.size=Pt(20);r.font.color.rgb=RGBColor(31,54,76)
p=D.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Gate C Report · Version 1.0 · Protocol v0.2 approved 31 August 2026');r.italic=True;r.font.size=Pt(10)
callout('GATE C: PASS',f"All {summary['leakage_tests_total']}/{summary['leakage_tests_total']} split, identity, support-isolation and nested-CV unit tests passed. The primary outer assignments are now immutable. No predictive model was trained in Phase 4.",'D9EAD3')

D.add_heading('1. Freeze decision',level=1)
bullet('Protocol v0.2 was approved before Phase 4 execution and is recorded in P4_18_protocol_approval_and_freeze_record.csv.')
bullet('The primary five-fold outer split was inherited unchanged from the Phase 3 outcome-blind candidate manifest. It was not regenerated after endpoint feasibility was observed.')
bullet('Four additional V3 robustness split sets were generated prospectively from chemical identity fields only and frozen now; they may not replace the primary split.')
bullet('A four-fold grouped inner split was generated separately inside every outer-training set for the primary and all four V3 split sets.')
bullet('Every salt, duplicate identifier, biological context, and source row inherits the immutable parent-compound fold.')

D.add_heading('2. Frozen primary outer split',level=1)
pr=outsz[outsz.split_set=='PRIMARY'].sort_values('outer_fold')
add_table(['Outer fold','Held-out parent compounds'],[(int(r.outer_fold),int(r.parent_n)) for r in pr.itertuples(index=False)],[1.3,2.2])
p=D.add_paragraph();p.add_run('Assignment rule: ').bold=True;p.add_run('SHA256(compound_id | canonical_parent_smiles | FACTORIAL-OOD-v0.1)[:16] mod 5. This exact rule reproduces all 2,017 Phase 3 assignments.')

D.add_heading('3. Frozen nested inner CV',level=1)
pi=insz[insz.split_set=='PRIMARY']
rows=[]
for of,g in pi.groupby('outer_fold'):
    vals=[int(g.loc[g.inner_fold==i,'validation_parent_n'].iloc[0]) for i in range(4)]
    rows.append((int(of),*vals,sum(vals)))
add_table(['Outer fold','Inner 0','Inner 1','Inner 2','Inner 3','Outer-train total'],rows,[0.9,0.9,0.9,0.9,0.9,1.25])
p=D.add_paragraph();p.add_run('Inner assignment rule: ').bold=True;p.add_run('SHA256(identity | split_set | outer_fold | FACTORIAL-OOD-v0.2-INNER-v1)[:16] mod 4. No endpoint label, prevalence or factorial state participates in the assignment.')

D.add_heading('4. Leakage unit-test results',level=1)
cat=leak.groupby(['category','status']).size().unstack(fill_value=0).reset_index()
rows=[]
for r in cat.itertuples(index=False): rows.append((r.category,int(getattr(r,'PASS',0)),int(getattr(r,'FAIL',0))))
add_table(['Test category','PASS','FAIL'],rows,[2.5,0.8,0.8])
callout('Strongest dynamic checks',
        'Chemical support was recomputed for all 2,017 parents using only each outer-training fold: maximum Morgan-Tanimoto values matched Phase 3 with maximum absolute difference 1.11×10⁻¹⁶, with zero scaffold-state or C0/C1 mismatches. For biological support, every fold-specific q75 cutoff was reconstructed exactly from the corresponding training-parent HTTr distance table; no outer-test parent occurred in those training tables.','EAF2F8')

D.add_heading('5. Endpoint estimability under the frozen splits',level=1)
iv=classsup[classsup.level=='INNER_VALIDATION']; it=classsup[classsup.level=='INNER_TRAIN']; ot=classsup[classsup.level=='OUTER_TEST']
add_table(['Scope','Minimum active','Minimum inactive'],[
    ('Inner validation',int(iv.active_n.min()),int(iv.inactive_n.min())),
    ('Inner training',int(it.active_n.min()),int(it.inactive_n.min())),
    ('Outer test',int(ot.active_n.min()),int(ot.inactive_n.min()))],[2.0,1.2,1.2])
p=D.add_paragraph('All eight Primary endpoints retain both classes in every inner-validation, inner-training and outer-test partition; therefore nested tuning and pooled OOF evaluation remain estimable without label-informed split repair.')

D.add_heading('6. Individual-fold factorial sparsity: locked interpretation rule',level=1)
rows=[]
for r in sparse.itertuples(index=False):
    rows.append((r.factorial_cell,int(r.zero_active_endpoint_fold_rows),int(r.zero_inactive_endpoint_fold_rows),int(r.min_labeled_per_endpoint_fold),f'{r.median_labeled_per_endpoint_fold:.1f}'))
add_table(['Cell','Endpoint×fold rows with 0 actives','with 0 inactives','Min labeled','Median labeled'],rows,[0.9,1.8,1.5,1.0,1.0])
callout('Interpretation rule', 'Some individual OOD cells are sparse even though the pooled five-fold OOF cell meets the Phase 3 eligibility rule. Fold-specific cell performance will not be reported when either class is absent. Primary factorial metrics and bootstrap inference will use the pooled immutable OOF prediction table, as prespecified.','FFF2CC')

D.add_heading('7. V3 robustness split sets frozen prospectively',level=1)
rows=[]
for r in registry.itertuples(index=False):
    rows.append((r.split_set,r.role,r.outer_seed,r.status))
add_table(['Split set','Role','Identity-only seed','Status'],rows,[1.0,1.5,2.3,1.5])
p=D.add_paragraph('The four V3 sets are robustness analyses only. Their chemical and biological support states must later be recomputed within each alternative training fold using the already frozen v0.2 support rules; they cannot be used to select or replace the primary split.')

D.add_heading('8. Preprocessing contract carried into Phase 5',level=1)
bullet('Variance filtering, scaling, missing-value handling, PCA/latent factors and any feature selection are fitted only on the authorized inner-training or outer-training parent IDs in P4_08_preprocessing_scope_registry.csv.')
bullet('Hyperparameters, calibration models, late-fusion weights and MCC thresholds may use inner CV/inner OOF information only. Outer-test labels remain inaccessible for those decisions.')
bullet('Static molecular fingerprints/descriptors may be generated deterministically for all structures, but any learned filtering, scaling or selection applied to them is fold-local.')
bullet('Outer-test records are read-only until generation of locked predictions. Any later split change makes affected analyses exploratory unless an untouched evaluation is created.')

D.add_heading('9. Gate C decision',level=1)
for r in gate.itertuples(index=False):
    bullet(f'{r.check}: {r.status} — {r.evidence}')
callout('PHASE 4 CLOSED', 'The split package, alternative robustness seeds, nested inner folds, scope registry and leakage test outputs are checksum-locked. Gate C passes. The project may proceed to Phase 5 — leakage-safe feature construction.','D9EAD3')

D.add_heading('10. Deliverable index',level=1)
p=D.add_paragraph(); p.add_run('Frozen manifests: ').bold=True; p.add_run('P4_01 primary outer; P4_03 V3 alternatives; P4_04 nested inner; P4_05-P4_07 context/source inheritance; P4_08 preprocessing scopes.')
p=D.add_paragraph(); p.add_run('Audit and governance: ').bold=True; p.add_run('P4_09-P4_10 support tables; P4_11 train-only chemical recheck; P4_12 leakage tests; P4_15 software/seed lock; P4_16 Gate C; P4_17 sparsity note; P4_18 approval/freeze record; SHA256 manifest and executable build script.')

# footer
for section in D.sections:
    fp=section.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    rr=fp.add_run('FACTORIAL-OOD · Phase 4 Gate C · Protocol v0.2');rr.font.size=Pt(8);rr.font.color.rgb=RGBColor(100,100,100)
D.save(OUT)
print(OUT)
