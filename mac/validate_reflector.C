// mac/validate_reflector.C
// Run with: root -l mac/validate_reflector.C
// This macro validates your data-driven reflector by comparing
// simulated PDF and CDF with thesis data

#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

// Function to load digitized data from a .txt file
std::vector<std::pair<double, double>> LoadDigitizedData(const char* filename) {
    std::vector<std::pair<double, double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        return data;
    }
    double theta, intensity;
    while (file >> theta >> intensity) {
        data.push_back({theta, intensity});
    }
    file.close();
    return data;
}

// Function to normalize a TGraph to unit area (probability density)
void NormalizeGraphToUnitArea(TGraph* g) {
    if (g->GetN() == 0) return;
    
    // Calculate area under the curve using trapezoidal rule
    double area = 0;
    for (int i = 0; i < g->GetN() - 1; i++) {
        double x1, y1, x2, y2;
        g->GetPoint(i, x1, y1);
        g->GetPoint(i+1, x2, y2);
        area += (y1 + y2) * (x2 - x1) / 2.0;
    }
    
    if (area > 0) {
        for (int i = 0; i < g->GetN(); i++) {
            double x, y;
            g->GetPoint(i, x, y);
            g->SetPoint(i, x, y / area);
        }
    }
}

// Function to convert TGraph to TH1D for consistent binning
TH1D* GraphToHistogram(TGraph* g, const char* name, const char* title, int nbins, double xmin, double xmax) {
    TH1D* h = new TH1D(name, title, nbins, xmin, xmax);
    
    // Interpolate graph onto histogram bins
    for (int i = 1; i <= nbins; i++) {
        double binCenter = h->GetBinCenter(i);
        double y = g->Eval(binCenter);
        h->SetBinContent(i, y);
    }
    
    return h;
}

// Function to compute CDF from PDF data
std::vector<std::pair<double, double>> ComputeCDF(const std::vector<std::pair<double, double>>& pdf) {
    std::vector<std::pair<double, double>> cdf;
    if (pdf.empty()) return cdf;
    
    // First, sort by theta
    auto sorted = pdf;
    std::sort(sorted.begin(), sorted.end());
    
    // Compute total integral
    double total = 0;
    for (size_t i = 0; i < sorted.size(); i++) {
        total += sorted[i].second;
    }
    
    // Build CDF
    double cum = 0;
    for (size_t i = 0; i < sorted.size(); i++) {
        cum += sorted[i].second / total;
        cdf.push_back({sorted[i].first, cum});
    }
    
    return cdf;
}

// Function to compute CDF from histogram
std::vector<std::pair<double, double>> HistogramToCDF(TH1D* h) {
    std::vector<std::pair<double, double>> cdf;
    int nBins = h->GetNbinsX();
    double total = h->GetEntries();
    if (total == 0) return cdf;
    
    double cum = 0;
    for (int i = 1; i <= nBins; i++) {
        cum += h->GetBinContent(i);
        double x = h->GetBinCenter(i);
        cdf.push_back({x, cum});
    }
    
    return cdf;
}

void validate_reflector() {
    
    // ============================================================
    // PART 0: Create output directory
    // ============================================================
    
    std::string outputDir = "validation_plots";
    gSystem->MakeDirectory(outputDir.c_str());
    std::cout << "Created directory: " << outputDir << std::endl;
    
    // ============================================================
    // PART 1: Load simulation output
    // ============================================================
    
    std::string rootFileName = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/G4d2o/data/Sim_D2ODetector011.root";
    
    TFile* file = TFile::Open(rootFileName.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "ERROR: Cannot open " << rootFileName << std::endl;
        return;
    }
    
    TTree* reflTree = (TTree*)file->Get("ReflectionTree");
    if (!reflTree) {
        std::cerr << "ERROR: ReflectionTree not found" << std::endl;
        file->Close();
        return;
    }
    
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "Loaded: " << rootFileName << std::endl;
    std::cout << "ReflectionTree entries: " << reflTree->GetEntries() << std::endl;
    std::cout << "=========================================================\n" << std::endl;
    
    // Create histograms for different incident angle ranges
    TH1D* h_incident_0_10   = new TH1D("h_incident_0_10", "Reflected Angle (0#circ-10#circ incident)", 36, -90, 90);
    TH1D* h_incident_10_20  = new TH1D("h_incident_10_20", "Reflected Angle (10#circ-20#circ incident)", 36, -90, 90);
    TH1D* h_incident_20_30  = new TH1D("h_incident_20_30", "Reflected Angle (20#circ-30#circ incident)", 36, -90, 90);
    TH1D* h_incident_30_40  = new TH1D("h_incident_30_40", "Reflected Angle (30#circ-40#circ incident)", 36, -90, 90);
    TH1D* h_incident_40_50  = new TH1D("h_incident_40_50", "Reflected Angle (40#circ-50#circ incident)", 36, -90, 90);
    TH1D* h_incident_50_60  = new TH1D("h_incident_50_60", "Reflected Angle (50#circ-60#circ incident)", 36, -90, 90);
    TH1D* h_incident_60_70  = new TH1D("h_incident_60_70", "Reflected Angle (60#circ-70#circ incident)", 36, -90, 90);
    TH1D* h_incident_70_80  = new TH1D("h_incident_70_80", "Reflected Angle (70#circ-80#circ incident)", 36, -90, 90);
    
    // Fill histograms
    double incident, reflected;
    reflTree->SetBranchAddress("incident_angle", &incident);
    reflTree->SetBranchAddress("reflected_angle", &reflected);
    
    Long64_t nEntries = reflTree->GetEntries();
    for (Long64_t i = 0; i < nEntries; i++) {
        reflTree->GetEntry(i);
        
        if (incident >= 0 && incident < 10)       h_incident_0_10->Fill(reflected);
        else if (incident >= 10 && incident < 20) h_incident_10_20->Fill(reflected);
        else if (incident >= 20 && incident < 30) h_incident_20_30->Fill(reflected);
        else if (incident >= 30 && incident < 40) h_incident_30_40->Fill(reflected);
        else if (incident >= 40 && incident < 50) h_incident_40_50->Fill(reflected);
        else if (incident >= 50 && incident < 60) h_incident_50_60->Fill(reflected);
        else if (incident >= 60 && incident < 70) h_incident_60_70->Fill(reflected);
        else if (incident >= 70 && incident < 80) h_incident_70_80->Fill(reflected);
    }
    
    // Normalize histograms to unit area (probability density)
    if (h_incident_0_10->GetEntries() > 0)   h_incident_0_10->Scale(1.0 / h_incident_0_10->GetEntries());
    if (h_incident_10_20->GetEntries() > 0)  h_incident_10_20->Scale(1.0 / h_incident_10_20->GetEntries());
    if (h_incident_20_30->GetEntries() > 0)  h_incident_20_30->Scale(1.0 / h_incident_20_30->GetEntries());
    if (h_incident_30_40->GetEntries() > 0)  h_incident_30_40->Scale(1.0 / h_incident_30_40->GetEntries());
    if (h_incident_40_50->GetEntries() > 0)  h_incident_40_50->Scale(1.0 / h_incident_40_50->GetEntries());
    if (h_incident_50_60->GetEntries() > 0)  h_incident_50_60->Scale(1.0 / h_incident_50_60->GetEntries());
    if (h_incident_60_70->GetEntries() > 0)  h_incident_60_70->Scale(1.0 / h_incident_60_70->GetEntries());
    if (h_incident_70_80->GetEntries() > 0)  h_incident_70_80->Scale(1.0 / h_incident_70_80->GetEntries());
    
    // Set axis titles and style for all histograms (no fill)
    TH1D* histograms[] = {h_incident_0_10, h_incident_10_20, h_incident_20_30, h_incident_30_40, 
                          h_incident_40_50, h_incident_50_60, h_incident_60_70, h_incident_70_80};
    for (auto h : histograms) {
        h->GetXaxis()->SetTitle("Reflected Angle (degrees)");
        h->GetYaxis()->SetTitle("Probability Density");
        h->SetLineColor(kBlue);
        h->SetLineWidth(2);
        h->SetFillStyle(0);  // No fill under the curve
    }
    
    // ============================================================
    // PART 2: Load thesis data
    // ============================================================
    
    std::string angularDir = "/home/manoja450/G4WithoutLeadSheilding/MODULE2/CUSTOMOPTICALMODULE2/G4d2o/angular_data";
    
    auto data_0   = LoadDigitizedData((angularDir + "/incident_0deg.txt").c_str());
    auto data_10  = LoadDigitizedData((angularDir + "/incident_10deg.txt").c_str());
    auto data_20  = LoadDigitizedData((angularDir + "/incident_20deg.txt").c_str());
    auto data_30  = LoadDigitizedData((angularDir + "/incident_30deg.txt").c_str());
    auto data_40  = LoadDigitizedData((angularDir + "/incident_40deg.txt").c_str());
    auto data_50  = LoadDigitizedData((angularDir + "/incident_50deg.txt").c_str());
    auto data_60  = LoadDigitizedData((angularDir + "/incident_60deg.txt").c_str());
    auto data_70  = LoadDigitizedData((angularDir + "/incident_70deg.txt").c_str());
    auto data_80  = LoadDigitizedData((angularDir + "/incident_80deg.txt").c_str());
    
    // Create PDF TGraphs
    TGraph* g_pdf_0 = new TGraph(data_0.size());
    TGraph* g_pdf_10 = new TGraph(data_10.size());
    TGraph* g_pdf_20 = new TGraph(data_20.size());
    TGraph* g_pdf_30 = new TGraph(data_30.size());
    TGraph* g_pdf_40 = new TGraph(data_40.size());
    TGraph* g_pdf_50 = new TGraph(data_50.size());
    TGraph* g_pdf_60 = new TGraph(data_60.size());
    TGraph* g_pdf_70 = new TGraph(data_70.size());
    TGraph* g_pdf_80 = new TGraph(data_80.size());
    
    for (size_t i = 0; i < data_0.size(); i++)  g_pdf_0->SetPoint(i, data_0[i].first, data_0[i].second);
    for (size_t i = 0; i < data_10.size(); i++) g_pdf_10->SetPoint(i, data_10[i].first, data_10[i].second);
    for (size_t i = 0; i < data_20.size(); i++) g_pdf_20->SetPoint(i, data_20[i].first, data_20[i].second);
    for (size_t i = 0; i < data_30.size(); i++) g_pdf_30->SetPoint(i, data_30[i].first, data_30[i].second);
    for (size_t i = 0; i < data_40.size(); i++) g_pdf_40->SetPoint(i, data_40[i].first, data_40[i].second);
    for (size_t i = 0; i < data_50.size(); i++) g_pdf_50->SetPoint(i, data_50[i].first, data_50[i].second);
    for (size_t i = 0; i < data_60.size(); i++) g_pdf_60->SetPoint(i, data_60[i].first, data_60[i].second);
    for (size_t i = 0; i < data_70.size(); i++) g_pdf_70->SetPoint(i, data_70[i].first, data_70[i].second);
    for (size_t i = 0; i < data_80.size(); i++) g_pdf_80->SetPoint(i, data_80[i].first, data_80[i].second);
    
    // Normalize thesis graphs to unit area (probability density)
    NormalizeGraphToUnitArea(g_pdf_0);
    NormalizeGraphToUnitArea(g_pdf_10);
    NormalizeGraphToUnitArea(g_pdf_20);
    NormalizeGraphToUnitArea(g_pdf_30);
    NormalizeGraphToUnitArea(g_pdf_40);
    NormalizeGraphToUnitArea(g_pdf_50);
    NormalizeGraphToUnitArea(g_pdf_60);
    NormalizeGraphToUnitArea(g_pdf_70);
    NormalizeGraphToUnitArea(g_pdf_80);
    
    // Convert thesis graphs to histograms for consistent binning
    TH1D* h_thesis_0 = GraphToHistogram(g_pdf_0, "h_thesis_0", "Thesis Data 0#circ", 36, -90, 90);
    TH1D* h_thesis_10 = GraphToHistogram(g_pdf_10, "h_thesis_10", "Thesis Data 10#circ", 36, -90, 90);
    TH1D* h_thesis_20 = GraphToHistogram(g_pdf_20, "h_thesis_20", "Thesis Data 20#circ", 36, -90, 90);
    TH1D* h_thesis_30 = GraphToHistogram(g_pdf_30, "h_thesis_30", "Thesis Data 30#circ", 36, -90, 90);
    TH1D* h_thesis_40 = GraphToHistogram(g_pdf_40, "h_thesis_40", "Thesis Data 40#circ", 36, -90, 90);
    TH1D* h_thesis_50 = GraphToHistogram(g_pdf_50, "h_thesis_50", "Thesis Data 50#circ", 36, -90, 90);
    TH1D* h_thesis_60 = GraphToHistogram(g_pdf_60, "h_thesis_60", "Thesis Data 60#circ", 36, -90, 90);
    TH1D* h_thesis_70 = GraphToHistogram(g_pdf_70, "h_thesis_70", "Thesis Data 70#circ", 36, -90, 90);
    TH1D* h_thesis_80 = GraphToHistogram(g_pdf_80, "h_thesis_80", "Thesis Data 80#circ", 36, -90, 90);
    
    // Style for thesis histograms (no fill, dashed lines)
    TH1D* thesis_hists[] = {h_thesis_0, h_thesis_10, h_thesis_20, h_thesis_30, 
                            h_thesis_40, h_thesis_50, h_thesis_60, h_thesis_70, h_thesis_80};
    for (auto h : thesis_hists) {
        h->SetLineColor(kRed);
        h->SetLineWidth(2);
        h->SetLineStyle(2);  // dashed
        h->SetFillStyle(0);   // No fill
        h->GetXaxis()->SetTitle("Reflected Angle (degrees)");
        h->GetYaxis()->SetTitle("Probability Density");
    }
    
    // ============================================================
    // PART 3: Compute CDFs from thesis data
    // ============================================================
    
    auto cdf_0   = ComputeCDF(data_0);
    auto cdf_10  = ComputeCDF(data_10);
    auto cdf_20  = ComputeCDF(data_20);
    auto cdf_30  = ComputeCDF(data_30);
    auto cdf_40  = ComputeCDF(data_40);
    auto cdf_50  = ComputeCDF(data_50);
    auto cdf_60  = ComputeCDF(data_60);
    auto cdf_70  = ComputeCDF(data_70);
    auto cdf_80  = ComputeCDF(data_80);
    
    // Create CDF TGraphs from thesis data
    TGraph* g_cdf_0 = new TGraph(cdf_0.size());
    TGraph* g_cdf_10 = new TGraph(cdf_10.size());
    TGraph* g_cdf_20 = new TGraph(cdf_20.size());
    TGraph* g_cdf_30 = new TGraph(cdf_30.size());
    TGraph* g_cdf_40 = new TGraph(cdf_40.size());
    TGraph* g_cdf_50 = new TGraph(cdf_50.size());
    TGraph* g_cdf_60 = new TGraph(cdf_60.size());
    TGraph* g_cdf_70 = new TGraph(cdf_70.size());
    TGraph* g_cdf_80 = new TGraph(cdf_80.size());
    
    for (size_t i = 0; i < cdf_0.size(); i++)  g_cdf_0->SetPoint(i, cdf_0[i].first, cdf_0[i].second);
    for (size_t i = 0; i < cdf_10.size(); i++) g_cdf_10->SetPoint(i, cdf_10[i].first, cdf_10[i].second);
    for (size_t i = 0; i < cdf_20.size(); i++) g_cdf_20->SetPoint(i, cdf_20[i].first, cdf_20[i].second);
    for (size_t i = 0; i < cdf_30.size(); i++) g_cdf_30->SetPoint(i, cdf_30[i].first, cdf_30[i].second);
    for (size_t i = 0; i < cdf_40.size(); i++) g_cdf_40->SetPoint(i, cdf_40[i].first, cdf_40[i].second);
    for (size_t i = 0; i < cdf_50.size(); i++) g_cdf_50->SetPoint(i, cdf_50[i].first, cdf_50[i].second);
    for (size_t i = 0; i < cdf_60.size(); i++) g_cdf_60->SetPoint(i, cdf_60[i].first, cdf_60[i].second);
    for (size_t i = 0; i < cdf_70.size(); i++) g_cdf_70->SetPoint(i, cdf_70[i].first, cdf_70[i].second);
    for (size_t i = 0; i < cdf_80.size(); i++) g_cdf_80->SetPoint(i, cdf_80[i].first, cdf_80[i].second);
    
    // Style for CDF graphs
    auto styleCDF = [](TGraph* g, int color, int style) {
        g->SetLineColor(color);
        g->SetLineWidth(2);
        g->SetLineStyle(style);
    };
    styleCDF(g_cdf_0, kRed, 2);
    styleCDF(g_cdf_10, kRed, 2);
    styleCDF(g_cdf_20, kRed, 2);
    styleCDF(g_cdf_30, kRed, 2);
    styleCDF(g_cdf_40, kRed, 2);
    styleCDF(g_cdf_50, kRed, 2);
    styleCDF(g_cdf_60, kRed, 2);
    styleCDF(g_cdf_70, kRed, 2);
    styleCDF(g_cdf_80, kRed, 2);
    
    // Compute CDFs from simulation histograms
    auto sim_cdf_0_10   = HistogramToCDF(h_incident_0_10);
    auto sim_cdf_10_20  = HistogramToCDF(h_incident_10_20);
    auto sim_cdf_20_30  = HistogramToCDF(h_incident_20_30);
    auto sim_cdf_30_40  = HistogramToCDF(h_incident_30_40);
    auto sim_cdf_40_50  = HistogramToCDF(h_incident_40_50);
    auto sim_cdf_50_60  = HistogramToCDF(h_incident_50_60);
    auto sim_cdf_60_70  = HistogramToCDF(h_incident_60_70);
    auto sim_cdf_70_80  = HistogramToCDF(h_incident_70_80);
    
    // Create CDF TGraphs from simulation
    TGraph* g_sim_cdf_0_10 = new TGraph(sim_cdf_0_10.size());
    TGraph* g_sim_cdf_10_20 = new TGraph(sim_cdf_10_20.size());
    TGraph* g_sim_cdf_20_30 = new TGraph(sim_cdf_20_30.size());
    TGraph* g_sim_cdf_30_40 = new TGraph(sim_cdf_30_40.size());
    TGraph* g_sim_cdf_40_50 = new TGraph(sim_cdf_40_50.size());
    TGraph* g_sim_cdf_50_60 = new TGraph(sim_cdf_50_60.size());
    TGraph* g_sim_cdf_60_70 = new TGraph(sim_cdf_60_70.size());
    TGraph* g_sim_cdf_70_80 = new TGraph(sim_cdf_70_80.size());
    
    for (size_t i = 0; i < sim_cdf_0_10.size(); i++)   g_sim_cdf_0_10->SetPoint(i, sim_cdf_0_10[i].first, sim_cdf_0_10[i].second);
    for (size_t i = 0; i < sim_cdf_10_20.size(); i++)  g_sim_cdf_10_20->SetPoint(i, sim_cdf_10_20[i].first, sim_cdf_10_20[i].second);
    for (size_t i = 0; i < sim_cdf_20_30.size(); i++)  g_sim_cdf_20_30->SetPoint(i, sim_cdf_20_30[i].first, sim_cdf_20_30[i].second);
    for (size_t i = 0; i < sim_cdf_30_40.size(); i++)  g_sim_cdf_30_40->SetPoint(i, sim_cdf_30_40[i].first, sim_cdf_30_40[i].second);
    for (size_t i = 0; i < sim_cdf_40_50.size(); i++)  g_sim_cdf_40_50->SetPoint(i, sim_cdf_40_50[i].first, sim_cdf_40_50[i].second);
    for (size_t i = 0; i < sim_cdf_50_60.size(); i++)  g_sim_cdf_50_60->SetPoint(i, sim_cdf_50_60[i].first, sim_cdf_50_60[i].second);
    for (size_t i = 0; i < sim_cdf_60_70.size(); i++)  g_sim_cdf_60_70->SetPoint(i, sim_cdf_60_70[i].first, sim_cdf_60_70[i].second);
    for (size_t i = 0; i < sim_cdf_70_80.size(); i++)  g_sim_cdf_70_80->SetPoint(i, sim_cdf_70_80[i].first, sim_cdf_70_80[i].second);
    
    // Style for simulation CDF graphs
    styleCDF(g_sim_cdf_0_10, kBlue, 1);
    styleCDF(g_sim_cdf_10_20, kBlue, 1);
    styleCDF(g_sim_cdf_20_30, kBlue, 1);
    styleCDF(g_sim_cdf_30_40, kBlue, 1);
    styleCDF(g_sim_cdf_40_50, kBlue, 1);
    styleCDF(g_sim_cdf_50_60, kBlue, 1);
    styleCDF(g_sim_cdf_60_70, kBlue, 1);
    styleCDF(g_sim_cdf_70_80, kBlue, 1);
    
    // ============================================================
    // PART 4: Create PDF plots in two separate canvases (preserving original layout)
    // ============================================================
    
    // Canvas 1: PDF plots (0°-50°) - 3x2 grid like validation_plots_1
    TCanvas* c_pdf1 = new TCanvas("c_pdf1", "PDF Validation (0-50 deg)", 1200, 800);
    c_pdf1->Divide(3, 2);
    
    c_pdf1->cd(1);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_0_10->Draw("HIST");
    h_thesis_0->Draw("HIST SAME");
    TLegend* leg1 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg1->AddEntry(h_incident_0_10, "Simulation (0#circ-10#circ)", "L");
    leg1->AddEntry(h_thesis_0, "Thesis Data (0#circ)", "L");
    leg1->Draw();
    
    c_pdf1->cd(2);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_10_20->Draw("HIST");
    h_thesis_10->Draw("HIST SAME");
    TLegend* leg2 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg2->AddEntry(h_incident_10_20, "Simulation (10#circ-20#circ)", "L");
    leg2->AddEntry(h_thesis_10, "Thesis Data (10#circ)", "L");
    leg2->Draw();
    
    c_pdf1->cd(3);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_20_30->Draw("HIST");
    h_thesis_20->Draw("HIST SAME");
    TLegend* leg3 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg3->AddEntry(h_incident_20_30, "Simulation (20#circ-30#circ)", "L");
    leg3->AddEntry(h_thesis_20, "Thesis Data (20#circ)", "L");
    leg3->Draw();
    
    c_pdf1->cd(4);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_30_40->Draw("HIST");
    h_thesis_30->Draw("HIST SAME");
    TLegend* leg4 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg4->AddEntry(h_incident_30_40, "Simulation (30#circ-40#circ)", "L");
    leg4->AddEntry(h_thesis_30, "Thesis Data (30#circ)", "L");
    leg4->Draw();
    
    c_pdf1->cd(5);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_40_50->Draw("HIST");
    h_thesis_40->Draw("HIST SAME");
    TLegend* leg5 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg5->AddEntry(h_incident_40_50, "Simulation (40#circ-50#circ)", "L");
    leg5->AddEntry(h_thesis_40, "Thesis Data (40#circ)", "L");
    leg5->Draw();
    
    c_pdf1->cd(6);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_50_60->Draw("HIST");
    h_thesis_50->Draw("HIST SAME");
    TLegend* leg6 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg6->AddEntry(h_incident_50_60, "Simulation (50#circ-60#circ)", "L");
    leg6->AddEntry(h_thesis_50, "Thesis Data (50#circ)", "L");
    leg6->Draw();
    
    c_pdf1->SaveAs((outputDir + "/pdf_validation_1.png").c_str());
    
    // Canvas 2: PDF plots (60°-80°) - 2x2 grid like validation_plots_2
    TCanvas* c_pdf2 = new TCanvas("c_pdf2", "PDF Validation (60-80 deg)", 1200, 800);
    c_pdf2->Divide(2, 2);
    
    c_pdf2->cd(1);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_60_70->Draw("HIST");
    h_thesis_60->Draw("HIST SAME");
    TLegend* leg7 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg7->AddEntry(h_incident_60_70, "Simulation (60#circ-70#circ)", "L");
    leg7->AddEntry(h_thesis_60, "Thesis Data (60#circ)", "L");
    leg7->Draw();
    
    c_pdf2->cd(2);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_70_80->Draw("HIST");
    h_thesis_70->Draw("HIST SAME");
    TLegend* leg8 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg8->AddEntry(h_incident_70_80, "Simulation (70#circ-80#circ)", "L");
    leg8->AddEntry(h_thesis_70, "Thesis Data (70#circ)", "L");
    leg8->Draw();
    
    c_pdf2->cd(3);
    gPad->SetGridx();
    gPad->SetGridy();
    h_incident_70_80->Draw("HIST");
    h_thesis_80->Draw("HIST SAME");
    TLegend* leg9 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg9->AddEntry(h_incident_70_80, "Simulation (70#circ-80#circ)", "L");
    leg9->AddEntry(h_thesis_80, "Thesis Data (80#circ)", "L");
    leg9->Draw();
    
    c_pdf2->SaveAs((outputDir + "/pdf_validation_2.png").c_str());
    
    // ============================================================
    // PART 5: Create CDF plots (all 9 in single canvas)
    // ============================================================
    
    TCanvas* c_cdf_all = new TCanvas("c_cdf_all", "CDF Validation - All Angles (Tyvek Reflector)", 1500, 1500);
    c_cdf_all->Divide(3, 3);
    
    c_cdf_all->cd(1);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_0_10->Draw("AL");
    g_sim_cdf_0_10->SetTitle("CDF: 0#circ-10#circ vs Thesis 0#circ");
    g_sim_cdf_0_10->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_0_10->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_0->Draw("L SAME");
    TLegend* leg_c1 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c1->AddEntry(g_sim_cdf_0_10, "Simulation (0#circ-10#circ)", "L");
    leg_c1->AddEntry(g_cdf_0, "Thesis Data (0#circ)", "L");
    leg_c1->Draw();
    
    c_cdf_all->cd(2);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_10_20->Draw("AL");
    g_sim_cdf_10_20->SetTitle("CDF: 10#circ-20#circ vs Thesis 10#circ");
    g_sim_cdf_10_20->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_10_20->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_10->Draw("L SAME");
    TLegend* leg_c2 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c2->AddEntry(g_sim_cdf_10_20, "Simulation (10#circ-20#circ)", "L");
    leg_c2->AddEntry(g_cdf_10, "Thesis Data (10#circ)", "L");
    leg_c2->Draw();
    
    c_cdf_all->cd(3);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_20_30->Draw("AL");
    g_sim_cdf_20_30->SetTitle("CDF: 20#circ-30#circ vs Thesis 20#circ");
    g_sim_cdf_20_30->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_20_30->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_20->Draw("L SAME");
    TLegend* leg_c3 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c3->AddEntry(g_sim_cdf_20_30, "Simulation (20#circ-30#circ)", "L");
    leg_c3->AddEntry(g_cdf_20, "Thesis Data (20#circ)", "L");
    leg_c3->Draw();
    
    c_cdf_all->cd(4);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_30_40->Draw("AL");
    g_sim_cdf_30_40->SetTitle("CDF: 30#circ-40#circ vs Thesis 30#circ");
    g_sim_cdf_30_40->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_30_40->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_30->Draw("L SAME");
    TLegend* leg_c4 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c4->AddEntry(g_sim_cdf_30_40, "Simulation (30#circ-40#circ)", "L");
    leg_c4->AddEntry(g_cdf_30, "Thesis Data (30#circ)", "L");
    leg_c4->Draw();
    
    c_cdf_all->cd(5);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_40_50->Draw("AL");
    g_sim_cdf_40_50->SetTitle("CDF: 40#circ-50#circ vs Thesis 40#circ");
    g_sim_cdf_40_50->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_40_50->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_40->Draw("L SAME");
    TLegend* leg_c5 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c5->AddEntry(g_sim_cdf_40_50, "Simulation (40#circ-50#circ)", "L");
    leg_c5->AddEntry(g_cdf_40, "Thesis Data (40#circ)", "L");
    leg_c5->Draw();
    
    c_cdf_all->cd(6);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_50_60->Draw("AL");
    g_sim_cdf_50_60->SetTitle("CDF: 50#circ-60#circ vs Thesis 50#circ");
    g_sim_cdf_50_60->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_50_60->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_50->Draw("L SAME");
    TLegend* leg_c6 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c6->AddEntry(g_sim_cdf_50_60, "Simulation (50#circ-60#circ)", "L");
    leg_c6->AddEntry(g_cdf_50, "Thesis Data (50#circ)", "L");
    leg_c6->Draw();
    
    c_cdf_all->cd(7);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_60_70->Draw("AL");
    g_sim_cdf_60_70->SetTitle("CDF: 60#circ-70#circ vs Thesis 60#circ");
    g_sim_cdf_60_70->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_60_70->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_60->Draw("L SAME");
    TLegend* leg_c7 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c7->AddEntry(g_sim_cdf_60_70, "Simulation (60#circ-70#circ)", "L");
    leg_c7->AddEntry(g_cdf_60, "Thesis Data (60#circ)", "L");
    leg_c7->Draw();
    
    c_cdf_all->cd(8);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_70_80->Draw("AL");
    g_sim_cdf_70_80->SetTitle("CDF: 70#circ-80#circ vs Thesis 70#circ");
    g_sim_cdf_70_80->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_70_80->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_70->Draw("L SAME");
    TLegend* leg_c8 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c8->AddEntry(g_sim_cdf_70_80, "Simulation (70#circ-80#circ)", "L");
    leg_c8->AddEntry(g_cdf_70, "Thesis Data (70#circ)", "L");
    leg_c8->Draw();
    
    c_cdf_all->cd(9);
    gPad->SetGridx();
    gPad->SetGridy();
    g_sim_cdf_70_80->Draw("AL");
    g_sim_cdf_70_80->SetTitle("CDF: 70#circ-80#circ vs Thesis 80#circ");
    g_sim_cdf_70_80->GetXaxis()->SetTitle("Reflected Angle (degrees)");
    g_sim_cdf_70_80->GetYaxis()->SetTitle("Cumulative Probability");
    g_cdf_80->Draw("L SAME");
    TLegend* leg_c9 = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_c9->AddEntry(g_sim_cdf_70_80, "Simulation (70#circ-80#circ)", "L");
    leg_c9->AddEntry(g_cdf_80, "Thesis Data (80#circ)", "L");
    leg_c9->Draw();
    
    c_cdf_all->SaveAs((outputDir + "/cdf_validation_all_angles.png").c_str());
    
    // ============================================================
    // PART 6: 2D Correlation Plot
    // ============================================================
    
    TCanvas* c2d = new TCanvas("c2d", "Incident vs Reflected 2D", 800, 600);
    TH2D* h2D = new TH2D("h2D", "Incident vs Reflected Angle (Tyvek Reflections);Incident Angle (degrees);Reflected Angle (degrees)", 
                         36, 0, 90, 36, -90, 90);
    
    reflTree->SetBranchAddress("incident_angle", &incident);
    reflTree->SetBranchAddress("reflected_angle", &reflected);
    for (Long64_t i = 0; i < nEntries; i++) {
        reflTree->GetEntry(i);
        h2D->Fill(incident, reflected);
    }
    
    h2D->Draw("COLZ");
    c2d->SaveAs((outputDir + "/incident_vs_reflected_2D.png").c_str());
    
    // ============================================================
    // PART 7: Statistics summary
    // ============================================================
    
    TCanvas* c_stats = new TCanvas("c_stats", "Reflection Statistics", 800, 600);
    TH1D* h_incident_all = new TH1D("h_incident_all", "Distribution of Incident Angles;Incident Angle (degrees);Entries", 36, 0, 90);
    TH1D* h_reflected_all = new TH1D("h_reflected_all", "Distribution of Reflected Angles;Reflected Angle (degrees);Entries", 36, -90, 90);
    
    reflTree->SetBranchAddress("incident_angle", &incident);
    reflTree->SetBranchAddress("reflected_angle", &reflected);
    for (Long64_t i = 0; i < nEntries; i++) {
        reflTree->GetEntry(i);
        h_incident_all->Fill(incident);
        h_reflected_all->Fill(reflected);
    }
    
    h_incident_all->SetLineColor(kBlue);
    h_incident_all->SetFillStyle(0);
    h_incident_all->SetLineWidth(2);
    h_reflected_all->SetLineColor(kRed);
    h_reflected_all->SetFillStyle(0);
    h_reflected_all->SetLineWidth(2);
    
    h_incident_all->Draw("HIST");
    h_reflected_all->Draw("HIST SAME");
    
    TLegend* leg_stats = new TLegend(0.65, 0.75, 0.9, 0.88);
    leg_stats->AddEntry(h_incident_all, "Incident Angles", "L");
    leg_stats->AddEntry(h_reflected_all, "Reflected Angles", "L");
    leg_stats->Draw();
    
    c_stats->SaveAs((outputDir + "/statistics.png").c_str());
    
    // ============================================================
    // PART 8: Summary
    // ============================================================
    
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "VALIDATION PLOTS SAVED TO: " << outputDir << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "PNG files created:" << std::endl;
    std::cout << "  PDF Validation:" << std::endl;
    std::cout << "    - pdf_validation_1.png (0#circ-50#circ incident, 3x2 grid)" << std::endl;
    std::cout << "    - pdf_validation_2.png (60#circ-80#circ incident, 2x2 grid)" << std::endl;
    std::cout << "  CDF Validation (all 9 angles in 3x3 grid):" << std::endl;
    std::cout << "    - cdf_validation_all_angles.png" << std::endl;
    std::cout << "  Additional plots:" << std::endl;
    std::cout << "    - incident_vs_reflected_2D.png" << std::endl;
    std::cout << "    - statistics.png" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Total reflections processed: " << nEntries << std::endl;
    std::cout << "=========================================================\n" << std::endl;
    
    // Cleanup
    delete c_pdf1;
    delete c_pdf2;
    delete c_cdf_all;
    delete c2d;
    delete c_stats;
    delete h2D;
    delete h_incident_all;
    delete h_reflected_all;
    for (auto h : histograms) delete h;
    for (auto h : thesis_hists) delete h;
    delete g_pdf_0; delete g_pdf_10; delete g_pdf_20; delete g_pdf_30; delete g_pdf_40;
    delete g_pdf_50; delete g_pdf_60; delete g_pdf_70; delete g_pdf_80;
    delete g_cdf_0; delete g_cdf_10; delete g_cdf_20; delete g_cdf_30; delete g_cdf_40;
    delete g_cdf_50; delete g_cdf_60; delete g_cdf_70; delete g_cdf_80;
    delete g_sim_cdf_0_10; delete g_sim_cdf_10_20; delete g_sim_cdf_20_30; delete g_sim_cdf_30_40;
    delete g_sim_cdf_40_50; delete g_sim_cdf_50_60; delete g_sim_cdf_60_70; delete g_sim_cdf_70_80;
    
    file->Close();
}
