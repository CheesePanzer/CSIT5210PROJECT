import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const root = 'D:/CSIT5210PROJECT';
const audit = JSON.parse(await fs.readFile(path.join(root, 'tmp/nn_handoff/audit_data.json'), 'utf8'));
if (!audit.passed) throw new Error('Result validation failed; do not publish summary.');
const out = path.join(root, 'outputs/01a18c34-6ab5-7d21-ad3d-bd0989a8bb6b');
const handoff = path.join(root, 'results/NN_handoff_2026-11-02');
await fs.mkdir(out, {recursive: true});
await fs.mkdir(handoff, {recursive: true});
const wb = Workbook.create();
const summary = wb.worksheets.add('Overall');
const grouped = wb.worksheets.add('Imbalance groups');
const results = wb.worksheets.add('Results');
const keys = ['dataset','selection_criterion','loss','imbalance_group','n_samples','n_classes',
  'accuracy','macro_f1','g_mean','ba','seed','device','source_report'];
const rows = audit.metrics.sort((a,b)=>a.dataset.localeCompare(b.dataset)||a.selection_criterion.localeCompare(b.selection_criterion)||a.loss.localeCompare(b.loss));
const end = rows.length+1;
results.getRange(`A1:M${end}`).values = [keys, ...rows.map(row=>keys.map(key=>row[key]))];
results.getRange(`G2:J${end}`).setNumberFormat('0.00%');
results.getRange(`E2:F${end}`).setNumberFormat('0');
results.getRange(`K2:K${end}`).setNumberFormat('0');
results.freezePanes.freezeRows(1);
results.getRange('A106').values = [['Source: raw_reports/*.json; one row is one dataset/loss/selection-criterion result.']];

const metrics = ['accuracy','macro_f1','g_mean','ba'];
const metricColumns = ['G','H','I','J'];
const means = [];
summary.getRange('B2').values = [['Neural network results']];
summary.getRange('B3').values = [['17 datasets; five-fold held-out predictions; single seed 42; equal dataset weights.']];
summary.getRange('B5:H5').values = [['Selection criterion','Loss','Datasets','Accuracy','Macro-F1','G-mean','BA']];
let r = 6;
for (const criterion of ['accuracy','macro_f1']) {
  for (const loss of ['ce','ovr','ovo']) {
    summary.getRange(`B${r}:C${r}`).values = [[criterion,loss]];
    summary.getRange(`D${r}`).formulas = [[`=COUNTIFS(Results!$B$2:$B$${end},B${r},Results!$C$2:$C$${end},C${r})`]];
    summary.getRange(`E${r}:H${r}`).formulas = [metricColumns.map(col=>
      `=AVERAGEIFS(Results!$${col}$2:$${col}$${end},Results!$B$2:$B$${end},$B${r},Results!$C$2:$C$${end},$C${r})`)];
    const members=rows.filter(row=>row.selection_criterion===criterion&&row.loss===loss);
    means.push({selection_criterion:criterion,loss,n_datasets:members.length,...Object.fromEntries(metrics.map(m=>[m,members.reduce((s,row)=>s+row[m],0)/members.length]))});
    r++;
  }
}
summary.getRange('E6:H11').setNumberFormat('0.00%');
summary.getRange('B14').values = [['OVO relative growth over OVR: ratio of the dataset means, not mean of per-dataset ratios.']];
summary.getRange('B15:F15').values = [['Selection criterion','Accuracy growth','Macro-F1 growth','G-mean growth','BA growth']];
for (const [row,criterion,ovr,ovo] of [[16,'accuracy',7,8],[17,'macro_f1',10,11]]) {
  summary.getRange(`B${row}`).values = [[criterion]];
  summary.getRange(`C${row}:F${row}`).formulas = [['E','F','G','H'].map(col=>`=IF(${col}${ovr}=0,"n.a.",${col}${ovo}/${col}${ovr}-1)`)];
}
summary.getRange('C16:F17').setNumberFormat('+0.00%;-0.00%;0.00%');

grouped.getRange('B2').values = [['Results by dataset imbalance']];
grouped.getRange('B3').values = [['Low IR < 3; medium 3 <= IR < 9; high IR >= 9. Dataset means, displayed as percentages.']];
grouped.getRange('B5:I5').values = [['Selection criterion','Imbalance group','Loss','Datasets','Accuracy','Macro-F1','G-mean','BA']];
const groupMeans=[];
r=6;
for (const criterion of ['accuracy','macro_f1']) {
  for (const group of ['low','medium','high']) {
    for (const loss of ['ce','ovr','ovo']) {
      grouped.getRange(`B${r}:D${r}`).values = [[criterion,group,loss]];
      grouped.getRange(`E${r}`).formulas = [[`=COUNTIFS(Results!$B$2:$B$${end},B${r},Results!$D$2:$D$${end},C${r},Results!$C$2:$C$${end},D${r})`]];
      grouped.getRange(`F${r}:I${r}`).formulas = [metricColumns.map(col=>
        `=AVERAGEIFS(Results!$${col}$2:$${col}$${end},Results!$B$2:$B$${end},$B${r},Results!$D$2:$D$${end},$C${r},Results!$C$2:$C$${end},$D${r})`)];
      const members=rows.filter(row=>row.selection_criterion===criterion&&row.loss===loss&&row.imbalance_group===group);
      groupMeans.push({selection_criterion:criterion,imbalance_group:group,loss,n_datasets:members.length,...Object.fromEntries(metrics.map(m=>[m,members.reduce((s,row)=>s+row[m],0)/members.length]))});
      r++;
    }
  }
}
grouped.getRange('F6:I23').setNumberFormat('0.00%');
grouped.freezePanes.freezeRows(5);

for (const [sheet,range] of [[summary,'B2:H20'],[grouped,'B2:I23'],[results,`A1:M${end}`]]) {
  sheet.showGridLines=false;
  sheet.getRange(range).format.font={name:'Arial',size:11,color:'#202936'};
  sheet.getRange(range).format.rowHeight=23;
  sheet.getRange(range).format.verticalAlignment='center';
}
for (const [sheet,range] of [[summary,'B5:H5'],[summary,'B15:F15'],[grouped,'B5:I5'],[results,'A1:M1']]) {
  sheet.getRange(range).format={fill:'#27384D',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:30,horizontalAlignment:'center',verticalAlignment:'center'};
}
for (const sheet of [summary,grouped]) {
  sheet.getRange('B2').format.font={name:'Arial',size:16,bold:true,color:'#202936'};
  sheet.getRange('B3').format.font={name:'Arial',size:10,italic:true,color:'#586879'};
  sheet.getRange('B5:C23').format.columnWidth=23;
  sheet.getRange('D5:I23').format.columnWidth=16;
}
summary.getRange('B14').format.font={name:'Arial',size:10,italic:true,color:'#586879'};
summary.getRange('B14').format.rowHeight=25;
summary.getRange('C5:C11').format.columnWidth=23;
summary.getRange('D5:H17').format.columnWidth=19;
grouped.getRange('D5:D23').format.columnWidth=14;
results.getRange(`A1:D${end}`).format.columnWidth=23;
results.getRange(`E1:L${end}`).format.columnWidth=14;
results.getRange(`M1:M${end}`).format.columnWidth=65;

wb.recalculate();
const calculated=summary.getRange('E6:H11').values;
means.forEach((row,i)=>metrics.forEach((metric,j)=>{
  if (Math.abs(calculated[i][j]-row[metric])>1e-10) throw new Error(`Formula result differs: ${i}/${metric}`);
}));
const groupedCalculated=grouped.getRange('F6:I23').values;
groupMeans.forEach((row,i)=>metrics.forEach((metric,j)=>{
  if (Math.abs(groupedCalculated[i][j]-row[metric])>1e-10) throw new Error(`Group formula differs: ${i}/${metric}`);
}));
// Change a disposable source value to verify formula dependencies, then restore it.
const original=results.getRange('G2').values[0][0];
results.getRange('G2').values=[[original+0.001]];
wb.recalculate();
if (Math.abs(summary.getRange('E6').values[0][0]-calculated[0][0]-0.001/17)>1e-10) throw new Error('Summary did not recalculate after source edit.');
results.getRange('G2').values=[[original]];
wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000});
console.log('FORMULA_ERROR_SCAN',errors.ndjson);
for (const [sheet,range,label] of [['Overall','B2:H17','overall'],['Imbalance groups','B2:I23','groups'],['Results','A1:L13','results']]) {
  const blob=await wb.render({sheetName:sheet,range,scale:1.4,format:'png'});
  await fs.writeFile(path.join(root,`tmp/nn_handoff/${label}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
const xlsxPath=path.join(out,'NN_results_summary.xlsx');
await output.save(xlsxPath);
await fs.copyFile(xlsxPath,path.join(handoff,'NN_results_summary.xlsx'));
function csv(records,columns) {
  const encode=value=>`"${String(value??'').replaceAll('"','""')}"`;
  return '\uFEFF'+[columns,...records.map(row=>columns.map(c=>row[c]))].map(row=>row.map(encode).join(',')).join('\r\n')+'\r\n';
}
for (const [name,data,columns] of [
  ['summary_metrics.csv',rows,keys],
  ['overall_means.csv',means,['selection_criterion','loss','n_datasets',...metrics]],
  ['imbalance_group_means.csv',groupMeans,['selection_criterion','imbalance_group','loss','n_datasets',...metrics]],
  ['fold_details.csv',audit.fold_details,Object.keys(audit.fold_details[0])],
  ['source_manifest.csv',audit.source_manifest,['path','bytes','sha256']],
]) await fs.writeFile(path.join(handoff,name),csv(data,columns),'utf8');
await fs.writeFile(path.join(handoff,'audit_report.json'),JSON.stringify(Object.fromEntries(Object.entries(audit).filter(([key])=>!['metrics','fold_details','source_manifest'].includes(key))),null,2),'utf8');
await fs.writeFile(path.join(root,'tmp/nn_handoff/summary_values.json'),JSON.stringify({means,groupMeans}), 'utf8');
console.log('SAVED',xlsxPath);
