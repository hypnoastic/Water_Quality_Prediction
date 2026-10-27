/** Editable academic deck. Run npm ci, then node build_deck.mjs from this folder. */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import pptxgen from 'pptxgenjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const data = JSON.parse(fs.readFileSync(path.join(here, 'results.json'), 'utf8'));
const team = JSON.parse(fs.readFileSync(path.join(root, 'submission/team.json'), 'utf8'));
const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Project 18 Team';
pptx.subject = 'A chronological comparison of XGBoost and a regularized MLP';
pptx.title = 'Water Quality Prediction via Gradient Boosting and Deep Neural Networks';
pptx.company = team.institution;
pptx.lang = 'en-IN';
pptx.theme = {headFontFace:'Georgia',bodyFontFace:'Arial',lang:'en-IN'};
const C = {paper:'F6F0E6',ink:'183C3E',muted:'59706E',teal:'087F8C',orange:'D57C47',line:'D8DCCF',white:'FFFFFF'};
const W=13.333333, H=7.5;
const sources = {
  dataset:'https://doi.org/10.6084/m9.figshare.27800394.v2',
  ccme:'https://ccme.ca/en/res/wqimanualen.pdf',
  xgb:'https://doi.org/10.1145/2939672.2939785',
  mlp:'https://www.deeplearningbook.org/contents/mlp.html',
  tabular:'https://arxiv.org/abs/2207.08815',
  uncertainty:'https://arxiv.org/abs/2107.07511',
};
const specs=[];
function text(s,content,x,y,w,h,opts={}) {
  s.addText(content,{x,y,w,h,fontFace:'Arial',fontSize:21,color:C.ink,margin:0,
    breakLine:false,valign:'top',paraSpaceAfterPt:12,...opts});
}
function base(title,num,note='') {
  const s=pptx.addSlide();s.background={color:C.paper};
  text(s,title,.65,.42,12.0,.85,{fontFace:'Georgia',fontSize:33,bold:true});
  text(s,'Water Quality Prediction',.65,7.12,8,.2,{fontSize:10,color:C.muted});
  text(s,`${num} / 12`,11.6,7.12,1.05,.2,{fontSize:10,color:C.muted,align:'right'});
  if(note)s.addNotes(note);
  specs.push({number:num,title});return s;
}
function heading(s,label,x,y,w=5.5){text(s,label,x,y,w,.4,{fontSize:22,bold:true,color:C.teal});}
function nativeTable(s,headers,rows,x,y,widths,opts={}) {
  s.addTable([headers.map(v=>({text:v,options:{bold:true,color:C.ink,fill:'E7E8D9'}})),...rows],{
    x,y,w:widths.reduce((a,b)=>a+b,0),colW:widths,
    rowH:.55,fontFace:'Arial',fontSize:17,color:C.ink,fill:C.paper,
    border:{type:'solid',color:C.line,pt:.5},margin:.1,
    autoPage:false,paraSpaceAfterPt:0,valign:'mid',...opts,
  });
}
function bar(s,series,x,y,w,h,opts={}) {
  s.addChart(pptx.ChartType.bar,series,{
    x,y,w,h,barDir:'bar',grouping:'clustered',showLegend:false,
    catAxisLabelFontFace:'Arial',catAxisLabelFontSize:17,
    valAxisLabelFontFace:'Arial',valAxisLabelFontSize:14,
    dataLabelFormatCode:'0.00',dataLabelPosition:'outEnd',showValue:true,
    dataLabelFontFace:'Arial',dataLabelFontSize:15,
    showTitle:false,showCatName:false,
    chartColors:[C.teal],showBorder:false,showShadow:false,valAxisMinVal:0,
    catAxisLineColor:C.line,valAxisLineColor:C.line,
    valGridLine:{color:C.line,width:.5},
    showSerName:false,
    ...opts,
  });
}

// 1. Cover. The photo is decorative and the evidence remains in native objects.
{
  const s=pptx.addSlide();s.background={color:C.paper};
  const photo=path.join(here,'assets/river.jpg');
  s.addImage({path:photo,x:8.05,y:0,w:W-8.05,h:H,sizing:{type:'cover',w:W-8.05,h:H}});
  text(s,'PROJECT 18',.7,.72,6,.35,{fontSize:15,bold:true,color:C.teal,charSpacing:3});
  text(s,'Water Quality\nPrediction',.7,1.45,7.05,1.9,{fontFace:'Georgia',fontSize:48,bold:true});
  text(s,'Gradient boosting and\ndeep neural networks',.74,3.65,6.6,.9,{fontSize:25,color:C.muted});
  text(s,'A chronological study of a calculated\nsurface-water index',.74,4.92,6.6,.8,{fontSize:21});
  text(s,`${team.team_id}  /  ${team.section}`, .74,6.35,6.6,.35,{fontSize:14,color:C.muted});
  text(s,'Team details are editable dummy values',.74,6.88,6.6,.25,{fontSize:11,color:C.muted});
  s.addNotes(`Project 18 compares a strong tabular baseline with an MLP on one continuous target. State the scope as source-index emulation from the start. Source: ${sources.dataset}. Decorative photo: David Kovalenko, https://unsplash.com/photos/birds-eye-view-of-river-and-green-trees-qYMa2-P-U0M, Unsplash license. The photograph does not depict a documented monitoring station. Team: ${team.members.map(m=>m.name+' ('+m.enrollment+')').join(', ')}.`);
  specs.push({number:1,title:'Water Quality Prediction'});
}
// 2. Research question.
{
  const s=base('Problem and research question',2,`The assignment permits regression of a documented index or classification under a defined class scheme. We choose only CCME_Values. ${sources.dataset}`);
  text(s,'Can a small neural network improve on a strong tree model when estimating a documented water-quality index?',.7,1.6,11.8,1.35,{fontSize:30,fontFace:'Georgia'});
  heading(s,'What the project delivers',.7,3.3);
  text(s,'One continuous target on a 0–100 scale\nA shared, chronological evaluation protocol\nModel comparison and feature explanations\nAn explicit audit of how the target was created',.7,3.95,7.8,2.0,{fontSize:22,breakLine:false,paraSpaceAfterPt:15});
  text(s,'Prediction inputs must be meaningful for the intended use.',9.1,3.6,3.4,1.4,{fontSize:27,bold:true,color:C.orange});
}
// 3. Data.
{
  const s=base('The dataset and eight measurements',3,`Source: ${sources.dataset}. Data audit: reports/data_audit.json. England supplies 75.3% of rows. Original data are already harmonized and imputed.`);
  text(s,'2,827,977 rows',.7,1.55,5.6,.65,{fontSize:38,bold:true,color:C.teal});
  text(s,'1940–2023  ·  Five countries',.74,2.3,6,.45,{fontSize:22,color:C.muted});
  const counts=data.data_audit.country_counts;
  bar(s,[{name:'Rows',labels:Object.keys(counts),values:Object.values(counts)}],.55,3.02,6.5,3.35,{dataLabelFormatCode:'#,##0',dataLabelFontSize:11,catAxisLabelFontSize:16,valAxisLabelFormatCode:'0.0,,"m"',valAxisMaxVal:2600000});
  heading(s,'Available inputs',7.65,1.72);
  text(s,'Ammonia and BOD\nDissolved oxygen\nOrthophosphate\npH and temperature\nNitrogen and nitrate',7.65,2.42,4.9,2.85,{fontSize:24,paraSpaceAfterPt:15});
  text(s,'No turbidity, conductivity or dissolved-solids columns are supplied.',7.65,5.68,4.85,.85,{fontSize:19,color:C.muted});
}
// 4. Target audit.
{
  const s=base('The derived-target audit',4,`Source notebook: notebooks/reference/Data_Processing_Harmonization.ipynb. Our formula audit is src/water_quality/data.py. Official definition: ${sources.ccme}. Maximum reconstruction error = 2.84e-14. The denominator 12 is inferred by exact agreement with every delivered label, while the source notebook uses len(df.columns).`);
  text(s,'100% of released labels',.7,1.6,11.8,.8,{fontSize:41,bold:true,color:C.orange});
  text(s,'reconstruct within 0.0000001 from the same eight inputs',.74,2.52,11.5,.65,{fontSize:25});
  heading(s,'Source-defined row-level calculation',.74,3.65,7.7);
  text(s,'Upper-only objectives\nF1 = F2 for an individual row\nDenominator = 12 for eight variables',.74,4.27,7.3,1.55,{fontSize:23,paraSpaceAfterPt:12});
  text(s,'This evaluates an index emulator.',8.55,3.77,3.9,1.25,{fontSize:29,bold:true,color:C.teal});
  text(s,'Official CCME WQI requires a defined assessment period and suitable objectives.',8.55,5.22,3.9,1.2,{fontSize:20,color:C.muted});
}
// 5. Preprocessing and dates.
{
  const s=base('Preprocessing and temporal evaluation',5,'Sources: reports/data_audit.json and reports/split_manifest.json. All transforms fit on the shared training sample. Historical sites can recur. The temporal holdout does not remove the structural dependence of the target on its inputs.');
  nativeTable(s,['Partition','Period','Available rows','Role'],[
    ['Training','1940–2018','2,478,761','300,000 shared rows'],
    ['Validation','2019','87,177','Tuning and model selection'],
    ['Calibration','2020','49,182','Residual interval radius'],
    ['Testing','2021–2023','202,886','Final frozen comparison'],
  ],.7,1.65,[1.85,2.15,2.7,5.2],{rowH:.61,fontSize:18});
  text(s,'Remove 9,971 exact duplicates. Preserve pollution extremes.',.74,5.05,11.8,.45,{fontSize:23,bold:true});
  text(s,'Flag invalid readings. Fit median imputation on training only.\nUse asinh and standard scaling for the MLP and Ridge.',.74,5.85,11.6,.95,{fontSize:22,color:C.muted,paraSpaceAfterPt:12});
}
// 6. Models.
{
  const s=base('Model design and bounded tuning',6,`Methods: ${sources.xgb}; ${sources.mlp}; ${sources.tabular}. Exact configuration: configs/default.json and src/water_quality/pipeline.py. Two candidates per learned family, one seed. The MLP reaches the 60-epoch budget, so full convergence is not established. Mean and Ridge are additional fixed baselines.`);
  heading(s,'XGBoost',.75,1.7);
  text(s,'Thresholds and tabular interactions',.75,2.35,5.55,.75,{fontSize:26,fontFace:'Georgia'});
  text(s,'Depths 4 and 6\nLearning rate 0.05\nUp to 1,000 boosting rounds\nValidation early stopping',.75,3.45,5.55,2.2,{fontSize:22,paraSpaceAfterPt:14});
  heading(s,'PyTorch MLP',7.25,1.7);
  text(s,'Learned nonlinear combinations',7.25,2.35,5.35,.75,{fontSize:26,fontFace:'Georgia'});
  text(s,'Hidden layers: 64/32 or 128/64\nReLU and dropout 0.1\nAdamW with weight decay\nUp to 60 training epochs',7.25,3.45,5.35,2.2,{fontSize:22,paraSpaceAfterPt:14});
  text(s,'Both use eight measurements plus eight missing/invalid flags.',.75,6.35,11.7,.45,{fontSize:21,color:C.muted});
}
// 7. Editable evidence.
{
  const s=base('XGBoost leads the individual models',7,'Measured sources: reports/metrics.csv and reports/validation_candidates.csv. All test scores use 202,886 held-out rows. Model family and blend weight use 2019 validation RMSE. The blend has a small RMSE improvement and worse MAE than XGBoost. No significance claim follows from one seed.');
  text(s,'Test RMSE in index points',.75,1.63,6.3,.45,{fontSize:22,color:C.muted});
  const learned=data.metrics.filter(r=>['XGBoost','MLP','Blend'].includes(r.model));
  bar(s,[{name:'RMSE',labels:learned.map(r=>r.model),values:learned.map(r=>r.rmse)}],.6,2.2,6.5,3.8,{valAxisMaxVal:1.1,dataLabelFormatCode:'0.000'});
  nativeTable(s,['Model','MAE','RMSE','R²'],data.metrics.map(r=>[r.model,r.mae.toFixed(3),r.rmse.toFixed(3),r.r2.toFixed(4)]),7.5,1.86,[1.7,1.05,1.05,1.25],{fontSize:16.5,rowH:.57});
  text(s,'Selected blend\n90% XGBoost + 10% MLP',7.55,5.75,4.9,1.0,{fontSize:23,bold:true,color:C.teal});
}
// 8. Native importance chart.
{
  const s=base('Orthophosphate drives model reliance',8,'Measured source: reports/permutation_importance.csv. Each measurement and its quality flag move together, with three shuffle repeats on 2,000 validation rows. Values below are for XGBoost. Native TreeSHAP provides a second explanation in reports/shap_importance.csv. These explain a calculated index and cannot establish environmental causality.');
  text(s,'Increase in validation MAE after shuffling',.75,1.6,11.8,.45,{fontSize:22,color:C.muted});
  const rows=data.importance.filter(r=>r.model==='XGBoost').sort((a,b)=>b.mae_increase-a.mae_increase).slice(0,6);
  bar(s,[{name:'MAE increase',labels:rows.map(r=>r.feature),values:rows.map(r=>r.mae_increase)}],.65,2.35,11.55,3.65,{valAxisMaxVal:10,dataLabelFormatCode:'0.00'});
  text(s,'These inputs directly enter the source formula. Attribution does not establish causation.',.75,6.4,11.8,.45,{fontSize:20,color:C.orange});
}
// 9. Subgroup uncertainty.
{
  const s=base('Interval coverage varies by index band',9,`Measured source: reports/subgroup_metrics.csv, selected Blend, source_index_band groups. Intervals use 2020 held-out absolute residuals with the finite-sample order statistic. ${sources.uncertainty}. Exchangeability is not established under temporal shift. Poor coverage=51.9%, Marginal=45.2%.`);
  text(s,'89.3% overall coverage',.75,1.58,11.8,.65,{fontSize:36,bold:true,color:C.teal});
  text(s,'Nominal coverage: 90%. The interval has one global residual radius.',.75,2.45,11.7,.5,{fontSize:22,color:C.muted});
  const rows=data.groups.filter(r=>r.model==='Blend'&&r.group_type==='source_index_band');
  bar(s,[{name:'Coverage (%)',labels:rows.map(r=>r.group),values:rows.map(r=>r.coverage*100)}],.6,3.1,8.0,3.35,{valAxisMaxVal:100,dataLabelFormatCode:'0.0"%"',chartColors:[C.orange]});
  text(s,'45.2%',9.25,3.45,3.25,.8,{fontSize:45,bold:true,color:C.orange});
  text(s,'coverage for\nMarginal rows',9.25,4.5,3.25,1.2,{fontSize:25});
}
// 10. Practical use, no dashboard graphics.
{
  const s=base('A reproducible project and live demo',10,'Code: app.py, src/water_quality/cli.py and src/water_quality/predict.py. Both demo and CLI load the saved artifacts, with train-fitted preprocessing. Input flags remain visible. Inputs must use the original units and headers. Artifacts are pickle based for preprocessing and should come only from trusted sources.');
  heading(s,'Use the trained models',.75,1.7,11.8);
  text(s,'streamlit run app.py',.75,2.48,11.6,.55,{fontFace:'Courier New',fontSize:26,color:C.teal});
  text(s,'Enter one sample or upload a CSV. Compare XGBoost, MLP and the blend.',.75,3.3,11.45,.85,{fontSize:25});
  text(s,'Saved models, executed notebooks and full test predictions accompany the reports.',.75,4.7,11.45,.9,{fontSize:25});
  text(s,'The demo estimates a research index. It makes no water-safety decision.',.75,6.15,11.45,.65,{fontSize:22,bold:true,color:C.orange});
}
// 11. Limitations.
{
  const s=base('What the results establish',11,'Sources: reports/report.md, docs/DATA.md and submission/Water_Quality_Project_Report.pdf. The evidence supports source-index emulation under the stated chronological protocol. It does not show independent water quality outcomes, a new site holdout, future-target forecasting or uniform uncertainty coverage.');
  heading(s,'What we verified',.75,1.75);
  text(s,'A documented target audit\nA fair tree-versus-MLP comparison\nMeasured explanations and coverage\nA reproducible inference workflow',.75,2.45,5.55,2.8,{fontSize:23,paraSpaceAfterPt:16});
  heading(s,'Research limits',7.2,1.75);
  text(s,'Target depends on the same inputs\nPrior imputation and unit questions\nUnequal country representation\nOne seed and bounded tuning',7.2,2.45,5.4,2.8,{fontSize:23,paraSpaceAfterPt:16});
  text(s,'A stronger study needs independent outcomes or a defined reduced sensor panel.',.75,6.1,11.6,.75,{fontSize:25,bold:true,color:C.teal});
}
// 12. Conclusion and references.
{
  const s=base('Conclusion and references',12,`Dataset: ${sources.dataset}. Official standard: ${sources.ccme}. XGBoost: ${sources.xgb}. Feedforward networks: ${sources.mlp}. Tabular benchmark: ${sources.tabular}. Uncertainty: ${sources.uncertainty}. Repository-organization reference: https://github.com/abhi-7-7/Flight_Delay_Prediction. Design inspiration: https://canva.link/jhwgwqkgeon4gbs. Team metadata intentionally contain dummy values at user request.`);
  text(s,'XGBoost is the stronger individual model.\nThe blend gives a small RMSE improvement.',.75,1.65,11.8,1.25,{fontSize:30,fontFace:'Georgia'});
  text(s,'When every formula input is available, the transparent source calculation is the best way to reproduce this exact index.',.75,3.15,11.6,1.0,{fontSize:25,color:C.teal});
  text(s,'Karim et al. (2025), Surface Water Quality Dataset, Figshare v2\nCCME (2017), Water Quality Index User Manual\nChen & Guestrin (2016), XGBoost, KDD\nGoodfellow et al. (2016), Deep Learning, chapter 6\nAngelopoulos & Bates (2021), Conformal Prediction, arXiv:2107.07511',.75,4.65,11.7,1.7,{fontSize:16.5,color:C.muted,paraSpaceAfterPt:9});
  text(s,'github.com/hypnoastic/Water_Quality_Prediction',.75,6.66,11.7,.3,{fontSize:16,bold:true});
}
await pptx.writeFile({fileName:path.join(here,'Water_Quality_Prediction.pptx')});
execFileSync(process.env.PYTHON || 'python3', [path.join(root,'scripts/repair_pptx_package.py'), path.join(here,'Water_Quality_Prediction.pptx')], {stdio:'inherit'});
fs.writeFileSync(path.join(here,'slide_manifest.json'),JSON.stringify({slideCount:specs.length,slides:specs},null,2)+'\n');
console.log(`Created ${specs.length} editable slides.`);
