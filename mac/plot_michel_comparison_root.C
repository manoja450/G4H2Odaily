// plot_michel_comparison_root.C
//
// Native ROOT version of michelanalysis.py's comparison plot, styled to
// match the thin-outline, mirrored-tick "HIST" look (no stat box, no
// markers, box frame with ticks on all four sides).
//
// Produces TWO PNGs:
//   1. MichelSpectrumComparison_ROOT.png         - MC scaled to Real Data's
//      total Counts (raw-count comparison).
//   2. MichelSpectrumComparison_ROOT_Normalized.png - both histograms
//      independently area-normalized to 1 (shape-only comparison).
//
// Run with:  root -l -q plot_michel_comparison_root.C

#include "TFile.h"
#include "TH1D.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TString.h"
#include "TAxis.h"
#include "TSystem.h"
#include "TMath.h"
#include "Rtypes.h"
#include <cstdio>

void plot_michel_comparison_root()
{
    // ============================================================
    // Same paths and threshold as michelanalysis.py
    // ============================================================
    TString base_dir  = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/NEXTmodify/G4d2o_DATA_DRIVEN_COPY";
    TString michel_path = base_dir + "/mac/all_histograms.root";
    TString sim_path     = base_dir + "/data/Sim_D2ODetector014.root";

    const double cut_value = 60.0;  // PE, same threshold as michelanalysis.py

    // Binning/line-width knobs, in case you want to tune the look to match a
    // reference plot exactly. REBIN_FACTOR merges N adjacent native bins into
    // one (1 = keep the source histogram's own bin width, the finest
    // available since it's already binned in the ROOT file). LINE_WIDTH is
    // ROOT's own width units (1 = thinnest).
    const int REBIN_FACTOR = 1;
    const int LINE_WIDTH = 3;   // <-- BOLD line width (was 1)

    // Output directory - same "PLOTS" convention as the python scripts
    TString plots_dir = TString(gSystem->WorkingDirectory()) + "/PLOTS";
    gSystem->mkdir(plots_dir, true);  // true = create parent dirs too; no-op if it already exists

    // ============================================================
    // Real data histogram
    // ============================================================
    TFile *fData = TFile::Open(michel_path);
    if (!fData || fData->IsZombie()) { printf("ERROR: could not open %s\n", michel_path.Data()); return; }

    TH1D *hData = (TH1D*)fData->Get("michel_energy");
    if (!hData) { printf("ERROR: 'michel_energy' not found in %s\n", michel_path.Data()); return; }
    hData = (TH1D*)hData->Clone("hData");
    hData->SetDirectory(0);
    if (REBIN_FACTOR > 1) hData->Rebin(REBIN_FACTOR);

    // Restrict to the same [cut_value, xmax] range used in the python script,
    // and compute the integral only over that range (matches michel_total in
    // the python script, which sums only the truncated array).
    int cutBin = hData->GetXaxis()->FindBin(cut_value);
    double xmax = hData->GetXaxis()->GetXmax();
    hData->GetXaxis()->SetRange(cutBin, hData->GetNbinsX());
    double dataIntegral = hData->Integral(cutBin, hData->GetNbinsX());

    // ============================================================
    // MC histogram from the tree, same (post-rebin) binning as hData so the
    // two overlay bin-for-bin.
    // ============================================================
    TFile *fSim = TFile::Open(sim_path);
    if (!fSim || fSim->IsZombie()) { printf("ERROR: could not open %s\n", sim_path.Data()); return; }

    TTree *tree = (TTree*)fSim->Get("Sim_Tree");
    if (!tree) { printf("ERROR: 'Sim_Tree' not found in %s\n", sim_path.Data()); return; }

    int nbinsFull = hData->GetXaxis()->GetNbins();
    double xmin = hData->GetXaxis()->GetXmin();
    // xmax already captured above, before SetRange() (SetRange doesn't change GetXmax()).

    // Confirmed from tree->Print(): "numHits" is a direct sub-branch of the
    // "eventData" branch (class simEvent, Int_t), matching what python read
    // as tree["eventData/numHits"].
    const char* candidates[] = { "eventData.numHits", "numHits" };
    const int nCandidates = sizeof(candidates) / sizeof(candidates[0]);
    TH1D *hMC = nullptr;
    bool filled = false;
    for (int i = 0; i < nCandidates; ++i) {
        gDirectory->Delete("hMC;*");
        TString drawExpr = Form("%s>>hMC(%d,%f,%f)", candidates[i], nbinsFull, xmin, xmax);
        TString cutExpr = Form("%s>=%f", candidates[i], cut_value);
        tree->Draw(drawExpr, cutExpr, "goff");
        TH1D *hTry = (TH1D*)gDirectory->Get("hMC");
        if (hTry && hTry->Integral() > 0) {
            printf(">>> Branch expression that worked: \"%s\" (%.0f entries)\n", candidates[i], hTry->Integral());
            hMC = hTry;
            hMC->SetDirectory(0);   // detach now, AFTER the fill, so it survives file closes below
            filled = true;
            break;
        }
    }

    if (!filled || !hMC) {
        printf("ERROR: neither candidate expression filled any entries. Re-run and paste\n");
        printf("the full console output (including any TTreeFormula errors above this line).\n");
        return;
    }

    double mcIntegral = hMC->Integral();

    // Bin width read directly off the (possibly rebinned) histogram, so the
    // axis label always reflects the actual binning in use - not hardcoded.
    double binWidth = hData->GetXaxis()->GetBinWidth(1);
    TString countsLabel = Form("Counts / %.0f PE", binWidth);
    TString normLabel = Form("Normalized Counts / %.0f PE", binWidth);

    // ============================================================
    // Build area-normalized clones BEFORE rescaling hMC to match Counts
    // below, so the normalized plot reflects each histogram's own shape
    // independent of the Counts-matching scale factor.
    // ============================================================
    TH1D *hDataNorm = (TH1D*)hData->Clone("hDataNorm");
    hDataNorm->Scale(1.0 / dataIntegral);
    TH1D *hMCNorm = (TH1D*)hMC->Clone("hMCNorm");
    hMCNorm->Scale(1.0 / mcIntegral);

    // Scale MC to the same total "Counts" as real data, so the two overlay
    // on an absolute Counts axis (this is what ROOT's Scale() does, and
    // matches the reference plot's un-normalized "Counts" y-axis).
    hMC->Scale(dataIntegral / mcIntegral);

    // ============================================================
    // Shared styling - bold and large
    // ============================================================
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    gStyle->SetTextFont(62);          // Bold Times for all text
    gStyle->SetTitleFont(62, "XYZ");  // Axis titles bold
    gStyle->SetLabelFont(62, "XYZ");  // Axis labels bold

    hData->SetLineColor(kBlue);   hData->SetLineWidth(LINE_WIDTH);
    hMC->SetLineColor(kRed);      hMC->SetLineWidth(LINE_WIDTH);
    hDataNorm->SetLineColor(kBlue); hDataNorm->SetLineWidth(LINE_WIDTH);
    hMCNorm->SetLineColor(kRed);    hMCNorm->SetLineWidth(LINE_WIDTH);

    // ============================================================
    // Plot 1: Counts (MC scaled to Real Data's total)
    // ============================================================
    TCanvas *c1 = new TCanvas("c1", "Michel Electron Spectrum", 1500, 1000);  // Larger canvas
    c1->SetTicks(1, 1);   // mirrored ticks on all four sides
    c1->SetLeftMargin(0.12);
    c1->SetBottomMargin(0.12);

    // Combined max across BOTH histograms, with headroom - fixes the
    // clipping you saw (ROOT sizes the frame off the FIRST histogram drawn,
    // so any taller bin in hMC beyond hData's own max gets cut off).
    double combinedMax1 = TMath::Max(hData->GetMaximum(), hMC->GetMaximum());
    hData->SetMaximum(combinedMax1 * 1.15);

    hData->GetXaxis()->SetRangeUser(0, 800);
    hData->GetXaxis()->SetTitle("PEs");
    hData->GetYaxis()->SetTitle(countsLabel);
    hData->GetXaxis()->SetTitleSize(0.06);   // Larger
    hData->GetYaxis()->SetTitleSize(0.06);
    hData->GetXaxis()->SetLabelSize(0.05);   // Larger
    hData->GetYaxis()->SetLabelSize(0.05);

    hData->Draw("HIST");
    hMC->Draw("HIST SAME");

    TLegend *leg1 = new TLegend(0.65, 0.75, 0.88, 0.88);
    leg1->SetBorderSize(0);
    leg1->SetFillStyle(0);
    leg1->SetTextSize(0.045);      // Larger legend text
    leg1->AddEntry(hData, "Real Data", "l");
    leg1->AddEntry(hMC, "G4 Monte Carlo", "l");
    leg1->Draw();

    c1->SaveAs(plots_dir + "/MichelSpectrumComparison_ROOT.png");

    // ============================================================
    // Plot 2: Normalized (each histogram independently area-normalized to 1)
    // ============================================================
    TCanvas *c2 = new TCanvas("c2", "Michel Electron Spectrum - Normalized", 1500, 1000);
    c2->SetTicks(1, 1);
    c2->SetLeftMargin(0.14);
    c2->SetBottomMargin(0.12);

    double combinedMax2 = TMath::Max(hDataNorm->GetMaximum(), hMCNorm->GetMaximum());
    hDataNorm->SetMaximum(combinedMax2 * 1.15);

    hDataNorm->GetXaxis()->SetRangeUser(0, 800);
    hDataNorm->GetXaxis()->SetTitle("PEs");
    hDataNorm->GetYaxis()->SetTitle(normLabel);
    hDataNorm->GetXaxis()->SetTitleSize(0.06);
    hDataNorm->GetYaxis()->SetTitleSize(0.06);
    hDataNorm->GetXaxis()->SetLabelSize(0.05);
    hDataNorm->GetYaxis()->SetLabelSize(0.05);

    hDataNorm->Draw("HIST");
    hMCNorm->Draw("HIST SAME");

    TLegend *leg2 = new TLegend(0.65, 0.75, 0.88, 0.88);
    leg2->SetBorderSize(0);
    leg2->SetFillStyle(0);
    leg2->SetTextSize(0.045);
    leg2->AddEntry(hDataNorm, "Real Data", "l");
    leg2->AddEntry(hMCNorm, "G4 Monte Carlo", "l");
    leg2->Draw();

    c2->SaveAs(plots_dir + "/MichelSpectrumComparison_ROOT_Normalized.png");

    // ============================================================
    printf("======================================================================\n");
    printf("Threshold                : %.0f PE\n", cut_value);
    printf("Bin width in use         : %.1f PE (REBIN_FACTOR=%d applied to the native binning)\n", binWidth, REBIN_FACTOR);
    printf("Real Data Integral (cut) : %.0f\n", dataIntegral);
    printf("MC Integral (cut, raw)   : %.0f\n", mcIntegral);
    printf("MC scale factor applied  : %.4f\n", dataIntegral / mcIntegral);
    printf("Plots saved to           : %s/\n", plots_dir.Data());
    printf("======================================================================\n");
}
