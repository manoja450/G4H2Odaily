// ============================================================================
// angular_analysis.cpp
// Tyvek Reflectivity Analysis - Gaussian + Lambertian Fit
// WITH Validation for Functions A, B, C (Data-Driven Reflector)
// Usage: ./angular_analysis [input_file] [output_directory]
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <map>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TF1.h>
#include <TLegend.h>
#include <TGraphErrors.h>
#include <TLine.h>
#include <TPaveText.h>
#include <TMath.h>
#include <TPaveStats.h>
#include <TText.h>

using namespace std;

// ============================================================================
// Helper function to get current working directory
// ============================================================================
string getCurrentDirectory() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        return string(cwd);
    }
    return ".";
}

// ============================================================================
// Helper function to create directory
// ============================================================================
bool createDirectory(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        #ifdef _WIN32
            return mkdir(path.c_str()) == 0;
        #else
            return mkdir(path.c_str(), 0755) == 0;
        #endif
    }
    return true;
}

// ============================================================================
// Helper function to get base filename without path and extension
// ============================================================================
string getBaseFilename(const string& filename) {
    string base = filename.substr(filename.find_last_of("/\\") + 1);
    return base.substr(0, base.find('.'));
}

// ============================================================================
// Fit Function: Gaussian + Lambertian (Equation 10 from thesis)
// ============================================================================
double gaussianLambertian(double *x, double *par) {
    double theta = x[0];
    double p1 = par[0];  // Lambertian amplitude
    double p2 = par[1];  // Gaussian amplitude
    double p3 = par[2];  // Gaussian center
    double p4 = par[3];  // Gaussian spread

    double theta_rad = theta * TMath::DegToRad();
    double lambertian = p1 * cos(theta_rad);
    double gaussian = p2 * exp(-pow(theta - p3, 2) / (2 * pow(p4, 2)));

    return lambertian + gaussian;
}

// ============================================================================
// Structure to hold fit results
// ============================================================================
struct FitResult {
    int phi;
    double p1, p1_err;
    double p2, p2_err;
    double p3, p3_err;
    double p4, p4_err;
    double ratio, ratio_err;
    int n_events;
    double chi2, ndof;
};

// ============================================================================
// Perform fit with constraints
// ============================================================================
TF1* performFit(TH1D* hist, int phi_target, FitResult& result) {
    double x_min = hist->GetXaxis()->GetXmin();
    double x_max = hist->GetXaxis()->GetXmax();

    TF1* fit = new TF1("fit", gaussianLambertian, x_min, x_max, 4);

    fit->SetParName(0, "p_{1} (Lambertian)");
    fit->SetParName(1, "p_{2} (Gaussian)");
    fit->SetParName(2, "p_{3} (Peak)");
    fit->SetParName(3, "p_{4} (#sigma)");

    double max_y = hist->GetMaximum();
    double peak_pos = hist->GetBinCenter(hist->GetMaximumBin());

    fit->SetParameter(0, max_y * 0.3);
    fit->SetParameter(1, max_y * 0.7);
    fit->SetParameter(2, peak_pos);
    fit->SetParameter(3, 20.0);

    fit->SetParLimits(0, 0, max_y * 2);
    fit->SetParLimits(1, 0, max_y * 2);
    fit->SetParLimits(2, -90, 90);
    fit->SetParLimits(3, 5, 60);

    hist->Fit(fit, "RQ");

    result.p1 = fit->GetParameter(0);
    result.p2 = fit->GetParameter(1);
    result.p3 = fit->GetParameter(2);
    result.p4 = fit->GetParameter(3);

    result.p1_err = fit->GetParError(0);
    result.p2_err = fit->GetParError(1);
    result.p3_err = fit->GetParError(2);
    result.p4_err = fit->GetParError(3);

    result.ratio = result.p2 / result.p1;
    if (result.p1 > 0 && result.p2 > 0) {
        result.ratio_err = result.ratio * sqrt(pow(result.p2_err/result.p2, 2) +
                                                pow(result.p1_err/result.p1, 2));
    } else {
        result.ratio_err = 0;
    }

    result.chi2 = fit->GetChisquare();
    result.ndof = fit->GetNDF();

    return fit;
}

// ============================================================================
// Create histogram from reflection angles
// ============================================================================
TH1D* createHistogram(const vector<double>& angles, int phi_target,
                      int bin_size, int margin) {
    if (angles.empty()) return nullptr;

    double data_min = *min_element(angles.begin(), angles.end());
    double data_max = *max_element(angles.begin(), angles.end());

    double x_min = data_min - margin;
    double x_max = data_max + margin;

    int n_bins = (int)ceil((x_max - x_min) / bin_size);
    if (n_bins < 1) n_bins = 1;

    TH1D* hist = new TH1D(Form("hist_%d", phi_target),
                          Form("Incident Angle = %d°", phi_target),
                          n_bins, x_min, x_max);

    for (double angle : angles) {
        hist->Fill(angle);
    }

    hist->Scale(1.0 / bin_size);

    hist->SetMarkerStyle(20);
    hist->SetMarkerSize(1.2);
    hist->SetMarkerColor(kBlue);
    hist->SetLineColor(kBlue);
    hist->SetLineWidth(1);
    hist->SetFillStyle(0);

    // Calculate Poisson errors
    for (int i = 1; i <= hist->GetNbinsX(); i++) {
        double content = hist->GetBinContent(i);
        double error = sqrt(content * bin_size) / bin_size;
        if (error == 0 && content > 0) {
            error = content * 0.03;
        }
        hist->SetBinError(i, error);
    }

    return hist;
}

// ============================================================================
// ==================== NEW VALIDATION FUNCTIONS (A, B, C) ====================
// ============================================================================

// ============================================================================
// Load thesis PDF data (Function A/B input)
// ============================================================================
bool loadThesisPDF(int incident_angle, vector<double>& theta, vector<double>& pdf) {
    string filename = Form("angular_data/incident_%ddeg.txt", incident_angle);
    ifstream file(filename);
    if (!file.is_open()) {
        // Try alternative path
        filename = Form("../angular_data/incident_%ddeg.txt", incident_angle);
        file.open(filename);
        if (!file.is_open()) {
            return false;
        }
    }
    
    theta.clear();
    pdf.clear();
    double t, intensity;
    while (file >> t >> intensity) {
        theta.push_back(t);
        pdf.push_back(intensity);
    }
    file.close();
    
    // Normalize to probability density
    if (theta.size() > 1) {
        double bin_width = theta[1] - theta[0];
        double sum = accumulate(pdf.begin(), pdf.end(), 0.0);
        for (auto& val : pdf) val /= (sum * bin_width);
    }
    
    return !theta.empty();
}

// ============================================================================
// Compute shape correlation between two histograms
// ============================================================================
double computeShapeCorrelation(const vector<double>& sim_hist, const vector<double>& ref_hist) {
    if (sim_hist.size() != ref_hist.size() || sim_hist.empty()) return 0.0;
    
    // Normalize
    double sum_sim = accumulate(sim_hist.begin(), sim_hist.end(), 0.0);
    double sum_ref = accumulate(ref_hist.begin(), ref_hist.end(), 0.0);
    
    vector<double> sim_norm(sim_hist.size()), ref_norm(ref_hist.size());
    for (size_t i = 0; i < sim_hist.size(); i++) {
        sim_norm[i] = (sum_sim > 0) ? sim_hist[i] / sum_sim : 0;
        ref_norm[i] = (sum_ref > 0) ? ref_hist[i] / sum_ref : 0;
    }
    
    // Compute correlation coefficient
    double mean_sim = accumulate(sim_norm.begin(), sim_norm.end(), 0.0) / sim_norm.size();
    double mean_ref = accumulate(ref_norm.begin(), ref_norm.end(), 0.0) / ref_norm.size();
    
    double cov = 0.0, var_sim = 0.0, var_ref = 0.0;
    for (size_t i = 0; i < sim_norm.size(); i++) {
        cov += (sim_norm[i] - mean_sim) * (ref_norm[i] - mean_ref);
        var_sim += pow(sim_norm[i] - mean_sim, 2);
        var_ref += pow(ref_norm[i] - mean_ref, 2);
    }
    
    if (var_sim * var_ref <= 0) return 0.0;
    return cov / sqrt(var_sim * var_ref);
}

// ============================================================================
// Compute chi-squared/ndf for validation
// ============================================================================
double computeChi2NDF(const vector<double>& sim_counts, const vector<double>& ref_counts) {
    double chi2 = 0.0;
    int ndof = 0;
    
    for (size_t i = 0; i < sim_counts.size(); i++) {
        if (ref_counts[i] > 0 && sim_counts[i] > 0) {
            chi2 += pow(sim_counts[i] - ref_counts[i], 2) / ref_counts[i];
            ndof++;
        }
    }
    
    ndof = max(ndof - 1, 1);
    return chi2 / ndof;
}

// ============================================================================
// FUNCTION C VALIDATION: Test sampling reproduces PDF for each incident angle
// ============================================================================
void validateFunctionC(TTree* tree, const vector<int>& target_angles, 
                       const string& output_dir, const string& file_prefix) {
    cout << "\n" << string(60, '=') << endl;
    cout << "FUNCTION C VALIDATION: Testing Sampling from PDF" << endl;
    cout << string(60, '=') << endl;
    
    // Get branch addresses
    Double_t incident_angle, reflected_angle;
    tree->SetBranchAddress("incident_angle", &incident_angle);
    tree->SetBranchAddress("reflected_angle", &reflected_angle);
    
    Long64_t nEntries = tree->GetEntries();
    
    // Create canvas for Function C validation plots
    TCanvas* canvas_func_c = new TCanvas("func_c", "Function C Validation", 1600, 1200);
    canvas_func_c->Divide(3, 3);
    canvas_func_c->SetLeftMargin(0.05);
    canvas_func_c->SetRightMargin(0.05);
    
    int subplot_idx = 1;
    
    for (int angle : target_angles) {
        // Collect simulated reflections for this incident angle
        vector<double> sim_reflections;
        for (Long64_t i = 0; i < nEntries; i++) {
            tree->GetEntry(i);
            if (fabs(incident_angle - angle) < 0.5) {
                sim_reflections.push_back(reflected_angle);
            }
        }
        
        if (sim_reflections.empty()) {
            cout << "  Incident " << angle << "°: No data" << endl;
            continue;
        }
        
        // Load thesis PDF
        vector<double> ref_theta, ref_pdf;
        if (!loadThesisPDF(angle, ref_theta, ref_pdf)) {
            cout << "  Incident " << angle << "°: No thesis data" << endl;
            continue;
        }
        
        // Create histogram from simulation
        int n_bins = ref_theta.size();
        double x_min = ref_theta.front();
        double x_max = ref_theta.back();
        
        TH1D* sim_hist = new TH1D(Form("sim_%d", angle), "", n_bins, x_min, x_max);
        for (double theta : sim_reflections) {
            sim_hist->Fill(theta);
        }
        
        // Get counts for chi2 calculation
        vector<double> sim_counts(n_bins), ref_counts(n_bins);
        for (int i = 1; i <= n_bins; i++) {
            sim_counts[i-1] = sim_hist->GetBinContent(i);
            ref_counts[i-1] = ref_pdf[i-1] * sim_reflections.size() * (ref_theta[1] - ref_theta[0]);
        }
        
        // Normalize to probability density for correlation
        sim_hist->Scale(1.0 / (sim_reflections.size() * (ref_theta[1] - ref_theta[0])));
        
        // Create reference histogram from thesis PDF
        TH1D* ref_hist = new TH1D(Form("ref_%d", angle), "", n_bins, x_min, x_max);
        for (int i = 1; i <= n_bins; i++) {
            ref_hist->SetBinContent(i, ref_pdf[i-1]);
        }
        
        // Compute validation metrics
        double correlation = computeShapeCorrelation(sim_counts, ref_counts);
        double chi2_ndf = computeChi2NDF(sim_counts, ref_counts);
        
        // Plot
        canvas_func_c->cd(subplot_idx++);
        gPad->SetGrid();
        
        sim_hist->SetMarkerStyle(20);
        sim_hist->SetMarkerSize(0.8);
        sim_hist->SetMarkerColor(kBlue);
        sim_hist->SetLineColor(kBlue);
        sim_hist->SetTitle(Form("Incident Angle = %d°", angle));
        sim_hist->Draw("E");
        
        ref_hist->SetLineColor(kRed);
        ref_hist->SetLineWidth(2);
        ref_hist->Draw("same");
        
        TLegend* leg = new TLegend(0.65, 0.70, 0.88, 0.88);
        leg->SetFillStyle(0);
        leg->SetBorderSize(0);
        leg->AddEntry(sim_hist, "Simulation (Function C)", "lp");
        leg->AddEntry(ref_hist, "Reference PDF (Thesis)", "l");
        leg->Draw();
        
        TPaveText* pt = new TPaveText(0.15, 0.78, 0.45, 0.88, "NDC");
        pt->SetFillStyle(0);
        pt->SetBorderSize(0);
        pt->SetTextSize(0.04);
        pt->AddText(Form("Correlation = %.4f", correlation));
        pt->AddText(Form("#chi^{2}/ndf = %.2f", chi2_ndf));
        pt->AddText(Form("n = %d", (int)sim_reflections.size()));
        pt->Draw();
        
        // Add pass/fail indicator
        TText* status = new TText();
        status->SetTextSize(0.08);
        status->SetTextColor((correlation > 0.99) ? kGreen : (correlation > 0.95 ? kOrange : kRed));
        status->DrawTextNDC(0.80, 0.92, (correlation > 0.99) ? "PASS" : (correlation > 0.95 ? "WARN" : "FAIL"));
        
        delete sim_hist;
        delete ref_hist;
    }
    
    string outfile = output_dir + "/" + file_prefix + "_function_c_validation.png";
    canvas_func_c->Print(outfile.c_str());
    cout << "✓ Function C validation plot saved: " << outfile << endl;
    
    delete canvas_func_c;
}

// ============================================================================
// FUNCTION A VALIDATION: Test interpolation between incident angles
// ============================================================================
void validateFunctionA(TTree* tree, const vector<int>& test_angles,
                       const string& output_dir, const string& file_prefix) {
    cout << "\n" << string(60, '=') << endl;
    cout << "FUNCTION A VALIDATION: Testing Interpolation" << endl;
    cout << string(60, '=') << endl;
    
    Double_t incident_angle, reflected_angle;
    tree->SetBranchAddress("incident_angle", &incident_angle);
    tree->SetBranchAddress("reflected_angle", &reflected_angle);
    
    Long64_t nEntries = tree->GetEntries();
    
    // Test angles that are NOT in the measured set (e.g., 5°, 15°, 25°, ...)
    vector<int> test_angles_to_check = {5, 15, 25, 35, 45, 55, 65, 75};
    
    TCanvas* canvas_func_a = new TCanvas("func_a", "Function A Validation", 1600, 1000);
    canvas_func_a->Divide(4, 2);
    
    int subplot_idx = 1;
    
    for (int angle : test_angles_to_check) {
        // Collect simulated reflections for this interpolated angle
        vector<double> sim_reflections;
        for (Long64_t i = 0; i < nEntries; i++) {
            tree->GetEntry(i);
            if (fabs(incident_angle - angle) < 0.5) {
                sim_reflections.push_back(reflected_angle);
            }
        }
        
        if (sim_reflections.size() < 100) {
            cout << "  Incident " << angle << "°: Only " << sim_reflections.size() 
                 << " samples, skipping" << endl;
            continue;
        }
        
        // Load bounding PDFs (Function A)
        int lower_angle = (angle / 10) * 10;
        int upper_angle = lower_angle + 10;
        
        vector<double> theta_low, pdf_low, theta_high, pdf_high;
        if (!loadThesisPDF(lower_angle, theta_low, pdf_low) ||
            !loadThesisPDF(upper_angle, theta_high, pdf_high)) {
            continue;
        }
        
        // Create interpolated PDF (Function A)
        double weight_high = (angle - lower_angle) / 10.0;
        double weight_low = 1.0 - weight_high;
        
        int n_bins = theta_low.size();
        double x_min = theta_low.front();
        double x_max = theta_low.back();
        
        TH1D* interp_hist = new TH1D(Form("interp_%d", angle), "", n_bins, x_min, x_max);
        for (int i = 1; i <= n_bins; i++) {
            double interp_val = weight_low * pdf_low[i-1] + weight_high * pdf_high[i-1];
            interp_hist->SetBinContent(i, interp_val);
        }
        
        // Create histogram from simulation
        TH1D* sim_hist = new TH1D(Form("sim_interp_%d", angle), "", n_bins, x_min, x_max);
        for (double theta : sim_reflections) {
            sim_hist->Fill(theta);
        }
        sim_hist->Scale(1.0 / (sim_reflections.size() * (theta_low[1] - theta_low[0])));
        
        // Compute correlation
        vector<double> sim_counts(n_bins), interp_counts(n_bins);
        for (int i = 1; i <= n_bins; i++) {
            sim_counts[i-1] = sim_hist->GetBinContent(i) * (theta_low[1] - theta_low[0]) * sim_reflections.size();
            interp_counts[i-1] = interp_hist->GetBinContent(i) * (theta_low[1] - theta_low[0]) * sim_reflections.size();
        }
        double correlation = computeShapeCorrelation(sim_counts, interp_counts);
        
        // Plot
        canvas_func_a->cd(subplot_idx++);
        gPad->SetGrid();
        
        sim_hist->SetMarkerStyle(20);
        sim_hist->SetMarkerSize(0.8);
        sim_hist->SetMarkerColor(kBlue);
        sim_hist->SetLineColor(kBlue);
        sim_hist->SetTitle(Form("Incident Angle = %d° (Interpolated)", angle));
        sim_hist->Draw("E");
        
        interp_hist->SetLineColor(kRed);
        interp_hist->SetLineWidth(2);
        interp_hist->Draw("same");
        
        TLegend* leg = new TLegend(0.65, 0.70, 0.88, 0.88);
        leg->SetFillStyle(0);
        leg->SetBorderSize(0);
        leg->AddEntry(sim_hist, "Simulation", "lp");
        leg->AddEntry(interp_hist, Form("Interpolated: %d° + %d°", lower_angle, upper_angle), "l");
        leg->Draw();
        
        TPaveText* pt = new TPaveText(0.15, 0.78, 0.45, 0.88, "NDC");
        pt->SetFillStyle(0);
        pt->SetBorderSize(0);
        pt->AddText(Form("Correlation = %.4f", correlation));
        pt->AddText(Form("n = %d", (int)sim_reflections.size()));
        pt->Draw();
        
        delete sim_hist;
        delete interp_hist;
    }
    
    string outfile = output_dir + "/" + file_prefix + "_function_a_validation.png";
    canvas_func_a->Print(outfile.c_str());
    cout << "✓ Function A validation plot saved: " << outfile << endl;
    
    delete canvas_func_a;
}

// ============================================================================
// FUNCTION B VALIDATION: Show continuous PDF interpolation
// ============================================================================
void validateFunctionB(const string& output_dir, const string& file_prefix) {
    cout << "\n" << string(60, '=') << endl;
    cout << "FUNCTION B VALIDATION: Continuous PDF Interpolation" << endl;
    cout << string(60, '=') << endl;
    
    vector<int> angles = {0, 30, 60, 80};
    
    TCanvas* canvas_func_b = new TCanvas("func_b", "Function B Validation", 1200, 1000);
    canvas_func_b->Divide(2, 2);
    
    int subplot_idx = 1;
    
    for (int angle : angles) {
        vector<double> theta, pdf;
        if (!loadThesisPDF(angle, theta, pdf)) {
            continue;
        }
        
        // Create continuous interpolation (0.5° resolution)
        double x_min = -90.0, x_max = 90.0;
        int n_fine = 361;  // 0.5° steps
        double step = (x_max - x_min) / n_fine;
        
        vector<double> fine_theta, fine_pdf;
        for (int i = 0; i <= n_fine; i++) {
            double t = x_min + i * step;
            fine_theta.push_back(t);
            
            // Linear interpolation
            double interp_val = 0.0;
            for (size_t j = 0; j < theta.size() - 1; j++) {
                if (t >= theta[j] && t <= theta[j+1]) {
                    double t_norm = (t - theta[j]) / (theta[j+1] - theta[j]);
                    interp_val = pdf[j] * (1 - t_norm) + pdf[j+1] * t_norm;
                    break;
                }
            }
            fine_pdf.push_back(interp_val);
        }
        
        canvas_func_b->cd(subplot_idx++);
        gPad->SetGrid();
        
        // Plot coarse points
        TGraph* coarse_graph = new TGraph(theta.size(), &theta[0], &pdf[0]);
        coarse_graph->SetMarkerStyle(20);
        coarse_graph->SetMarkerSize(1.5);
        coarse_graph->SetMarkerColor(kRed);
        coarse_graph->SetTitle(Form("Incident Angle = %d° (Function B)", angle));
        coarse_graph->Draw("AP");
        
        // Plot continuous line
        TGraph* fine_graph = new TGraph(fine_theta.size(), &fine_theta[0], &fine_pdf[0]);
        fine_graph->SetLineColor(kBlue);
        fine_graph->SetLineWidth(2);
        fine_graph->Draw("L same");
        
        TLegend* leg = new TLegend(0.65, 0.75, 0.88, 0.88);
        leg->SetFillStyle(0);
        leg->SetBorderSize(0);
        leg->AddEntry(coarse_graph, "Coarse data (5° bins)", "lp");
        leg->AddEntry(fine_graph, "Continuous interpolation (0.5°)", "l");
        leg->Draw();
        
        coarse_graph->GetXaxis()->SetTitle("Reflection Angle (degrees)");
        coarse_graph->GetYaxis()->SetTitle("Probability Density");
        
        delete coarse_graph;
        delete fine_graph;
    }
    
    string outfile = output_dir + "/" + file_prefix + "_function_b_validation.png";
    canvas_func_b->Print(outfile.c_str());
    cout << "✓ Function B validation plot saved: " << outfile << endl;
    
    delete canvas_func_b;
}

// ============================================================================
// 2D Correlation Plot (Incident vs Reflected)
// ============================================================================
void create2DCorrelationPlot(TTree* tree, const string& output_dir, const string& file_prefix) {
    cout << "\n" << string(60, '=') << endl;
    cout << "Creating 2D Correlation Plot (Incident vs Reflected)" << endl;
    cout << string(60, '=') << endl;
    
    Double_t incident_angle, reflected_angle;
    tree->SetBranchAddress("incident_angle", &incident_angle);
    tree->SetBranchAddress("reflected_angle", &reflected_angle);
    
    Long64_t nEntries = tree->GetEntries();
    
    // Fill arrays for 2D histogram
    vector<double> incident_vals, reflected_vals;
    incident_vals.reserve(nEntries);
    reflected_vals.reserve(nEntries);
    
    for (Long64_t i = 0; i < nEntries; i++) {
        tree->GetEntry(i);
        incident_vals.push_back(incident_angle);
        reflected_vals.push_back(reflected_angle);
    }
    
    TCanvas* canvas_2d = new TCanvas("canvas_2d", "2D Correlation", 1000, 900);
    
    TH2D* h2d = new TH2D("h2d", "Incident vs Reflected Angles", 
                          36, 0, 90, 36, -90, 90);
    
    for (size_t i = 0; i < incident_vals.size(); i++) {
        h2d->Fill(incident_vals[i], reflected_vals[i]);
    }
    
    h2d->SetStats(0);
    h2d->GetXaxis()->SetTitle("Incident Angle (degrees)");
    h2d->GetYaxis()->SetTitle("Reflected Angle (degrees)");
    h2d->Draw("COLZ");
    
    // Draw specular line
    TLine* spec_line = new TLine(0, 0, 90, 90);
    spec_line->SetLineColor(kRed);
    spec_line->SetLineStyle(2);
    spec_line->SetLineWidth(2);
    spec_line->Draw("same");
    
    TLine* spec_line_neg = new TLine(0, 0, 90, -90);
    spec_line_neg->SetLineColor(kRed);
    spec_line_neg->SetLineStyle(2);
    spec_line_neg->SetLineWidth(2);
    spec_line_neg->Draw("same");
    
    // Add text
    TPaveText* pt = new TPaveText(0.65, 0.80, 0.88, 0.88, "NDC");
    pt->SetFillStyle(0);
    pt->SetBorderSize(0);
    pt->AddText(Form("Total reflections: %lld", nEntries));
    pt->Draw();
    
    string outfile = output_dir + "/" + file_prefix + "_2d_correlation.png";
    canvas_2d->Print(outfile.c_str());
    cout << "✓ 2D correlation plot saved: " << outfile << endl;
    
    delete canvas_2d;
    delete h2d;
}

// ============================================================================
// Print thesis-style table
// ============================================================================
void printThesisTable(const vector<FitResult>& results, ofstream& logfile) {
    cout << "\n" << string(80, '=') << endl;
    cout << "THESIS-STYLE TABLE - Ratio of Gaussian/Lambertian Components" << endl;
    cout << string(80, '=') << endl;
    cout << setw(25) << "Angle of incidence φ"
         << setw(20) << "Φ₂/Φₗ (Simulation)"
         << setw(15) << "Error" << endl;
    cout << string(60, '-') << endl;

    logfile << "\n" << string(80, '=') << endl;
    logfile << "THESIS-STYLE TABLE - Ratio of Gaussian/Lambertian Components" << endl;
    logfile << string(80, '=') << endl;
    logfile << setw(25) << "Angle of incidence φ"
            << setw(20) << "Φ₂/Φₗ (Simulation)"
            << setw(15) << "Error" << endl;
    logfile << string(60, '-') << endl;

    for (const auto& r : results) {
        cout << setw(23) << r.phi << "°"
             << setw(18) << fixed << setprecision(3) << r.ratio
             << setw(12) << "± " << setprecision(3) << r.ratio_err << endl;

        logfile << setw(23) << r.phi << "°"
                << setw(18) << fixed << setprecision(3) << r.ratio
                << setw(12) << "± " << setprecision(3) << r.ratio_err << endl;
    }
    cout << string(80, '=') << endl;
    logfile << string(80, '=') << endl;
}

// ============================================================================
// Print comparison with thesis data
// ============================================================================
void printComparisonTable(const vector<FitResult>& results, ofstream& logfile) {
    vector<double> thesis_data = {0.93, 1.26, 0.859, 0.375, 0.517,
                                   0.527, 0.643, 0.963, 1.26};

    cout << "\n" << string(80, '=') << endl;
    cout << "COMPARISON WITH THESIS DATA (Chavarria 2007)" << endl;
    cout << string(80, '=') << endl;
    cout << setw(25) << "Angle of incidence φ"
         << setw(18) << "Simulation"
         << setw(18) << "Thesis"
         << setw(15) << "Difference" << endl;
    cout << string(76, '-') << endl;

    logfile << "\n" << string(80, '=') << endl;
    logfile << "COMPARISON WITH THESIS DATA (Chavarria 2007)" << endl;
    logfile << string(80, '=') << endl;
    logfile << setw(25) << "Angle of incidence φ"
            << setw(18) << "Simulation"
            << setw(18) << "Thesis"
            << setw(15) << "Difference" << endl;
    logfile << string(76, '-') << endl;

    for (size_t i = 0; i < results.size(); i++) {
        double diff = results[i].ratio - thesis_data[i];
        cout << setw(23) << results[i].phi << "°"
             << setw(15) << fixed << setprecision(3) << results[i].ratio
             << setw(15) << thesis_data[i]
             << setw(12) << showpos << fixed << setprecision(3) << diff << noshowpos << endl;

        logfile << setw(23) << results[i].phi << "°"
                << setw(15) << fixed << setprecision(3) << results[i].ratio
                << setw(15) << thesis_data[i]
                << setw(12) << showpos << fixed << setprecision(3) << diff << noshowpos << endl;
    }
    cout << string(80, '=') << endl;
    logfile << string(80, '=') << endl;
}

// ============================================================================
// Create comparison plot
// ============================================================================
void createComparisonPlot(const vector<FitResult>& results, const string& output_dir,
                          const string& file_prefix) {
    if (results.empty()) return;

    vector<double> thesis_data = {0.93, 1.26, 0.859, 0.375, 0.517,
                                   0.527, 0.643, 0.963, 1.26};

    vector<double> phi_vals, ratio_vals, ratio_errs;
    for (const auto& r : results) {
        phi_vals.push_back(r.phi);
        ratio_vals.push_back(r.ratio);
        ratio_errs.push_back(r.ratio_err);
    }

    TCanvas* canvas = new TCanvas("comp_canvas", "Comparison with Thesis", 1000, 700);
    canvas->SetGrid();
    canvas->SetLeftMargin(0.12);
    canvas->SetBottomMargin(0.12);

    TGraphErrors* sim_graph = new TGraphErrors(phi_vals.size(), &phi_vals[0], &ratio_vals[0],
                                                nullptr, &ratio_errs[0]);
    sim_graph->SetMarkerStyle(20);
    sim_graph->SetMarkerSize(1.8);
    sim_graph->SetMarkerColor(kBlue);
    sim_graph->SetLineColor(kBlue);
    sim_graph->SetLineWidth(2);

    TGraph* thesis_graph = new TGraph(phi_vals.size(), &phi_vals[0], &thesis_data[0]);
    thesis_graph->SetMarkerStyle(21);
    thesis_graph->SetMarkerSize(1.8);
    thesis_graph->SetMarkerColor(kRed);
    thesis_graph->SetLineColor(kRed);
    thesis_graph->SetLineWidth(2);

    sim_graph->SetTitle("Comparison of Gaussian/Lambertian Component Ratios");
    sim_graph->GetXaxis()->SetTitle("Angle of Incidence #phi (degrees)");
    sim_graph->GetYaxis()->SetTitle("#Phi_{2}/#Phi_{L} (Gaussian/Lambertian Ratio)");
    sim_graph->GetXaxis()->SetTitleSize(0.045);
    sim_graph->GetYaxis()->SetTitleSize(0.045);
    sim_graph->GetXaxis()->SetLabelSize(0.04);
    sim_graph->GetYaxis()->SetLabelSize(0.04);
    sim_graph->GetXaxis()->SetLimits(-2, 82);

    double y_max = max(*max_element(ratio_vals.begin(), ratio_vals.end()),
                      *max_element(thesis_data.begin(), thesis_data.end())) * 1.1;
    sim_graph->SetMinimum(0);
    sim_graph->SetMaximum(y_max);

    sim_graph->Draw("APL");
    thesis_graph->Draw("PL same");

    TLegend* legend = new TLegend(0.65, 0.75, 0.88, 0.88);
    legend->SetFillStyle(0);
    legend->SetBorderSize(0);
    legend->SetTextSize(0.04);
    legend->AddEntry(sim_graph, "Simulation (Data-Driven Reflector)", "lp");
    legend->AddEntry(thesis_graph, "Thesis Data (Chavarria 2007)", "lp");
    legend->Draw();

    TLine* line = new TLine(-2, 1, 82, 1);
    line->SetLineStyle(9);
    line->SetLineColor(kGray);
    line->SetLineWidth(1);
    line->Draw();

    string outfile_pdf = output_dir + "/" + file_prefix + "_ratio_comparison.png";
    canvas->Print(outfile_pdf.c_str());

    cout << "✓ Comparison plot saved as: " << outfile_pdf << endl;

    delete canvas;
}

// ============================================================================
// Process a single ROOT file
// ============================================================================
void processFile(const string& filename, const string& output_dir) {
    cout << "\n" << string(60, '#') << endl;
    cout << "# Processing: " << filename << endl;
    cout << string(60, '#') << endl;

    string file_prefix = getBaseFilename(filename);

    string log_filename = output_dir + "/" + file_prefix + "_analysis.log";
    ofstream logfile(log_filename.c_str());

    logfile << "Tyvek Reflectivity Analysis - " << filename << endl;
    logfile << "================================================" << endl;
    logfile << "Analysis started: " << __DATE__ << " " << __TIME__ << endl;

    TFile* file = TFile::Open(filename.c_str());
    if (!file || file->IsZombie()) {
        cerr << "ERROR: Cannot open file " << filename << endl;
        logfile << "ERROR: Cannot open file " << filename << endl;
        logfile.close();
        return;
    }

    TTree* tree = (TTree*)file->Get("ReflectionTree");
    if (!tree) {
        cerr << "ERROR: Cannot find ReflectionTree in file" << endl;
        logfile << "ERROR: Cannot find ReflectionTree in file" << endl;
        file->Close();
        logfile.close();
        return;
    }

    // Use Double_t for branch types
    Double_t incident_angle, reflected_angle;
    Int_t event_id, reflection_id;

    tree->SetBranchAddress("incident_angle", &incident_angle);
    tree->SetBranchAddress("reflected_angle", &reflected_angle);
    tree->SetBranchAddress("event_id", &event_id);
    tree->SetBranchAddress("reflection_id", &reflection_id);

    Long64_t nEntries = tree->GetEntries();
    cout << "Total reflections recorded: " << nEntries << endl;
    logfile << "Total reflections recorded: " << nEntries << endl;

    // Count unique events
    vector<int> events;
    for (Long64_t i = 0; i < nEntries && i < 10000000; i++) {
        tree->GetEntry(i);
        events.push_back(event_id);
    }
    sort(events.begin(), events.end());
    events.erase(unique(events.begin(), events.end()), events.end());
    cout << "Unique events with reflections: " << events.size() << endl;
    logfile << "Unique events with reflections: " << events.size() << endl;

    vector<int> target_angles = {0, 10, 20, 30, 40, 50, 60, 70, 80};
    int tolerance = 3;
    int bin_size = 5;
    int margin = 5;

    vector<FitResult> results;

    TCanvas* canvas = new TCanvas("canvas", "Tyvek Reflectivity Analysis", 1800, 1500);
    canvas->Divide(3, 3);
    canvas->SetLeftMargin(0.05);
    canvas->SetRightMargin(0.05);

    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    gStyle->SetGridStyle(3);

    cout << "\n" << string(60, '-') << endl;
    cout << "Fitting Results:" << endl;
    cout << string(60, '-') << endl;

    logfile << "\n" << string(60, '-') << endl;
    logfile << "Fitting Results:" << endl;
    logfile << string(60, '-') << endl;

    for (size_t idx = 0; idx < target_angles.size(); idx++) {
        int phi_target = target_angles[idx];

        vector<double> reflection_angles;

        for (Long64_t i = 0; i < nEntries; i++) {
            tree->GetEntry(i);
            if (fabs(incident_angle - phi_target) <= tolerance) {
                reflection_angles.push_back(reflected_angle);
            }
        }

        if (reflection_angles.empty()) {
            cout << "φ = " << setw(2) << phi_target << "°: NO DATA" << endl;
            logfile << "φ = " << setw(2) << phi_target << "°: NO DATA" << endl;
            continue;
        }

        cout << "φ = " << setw(2) << phi_target << "°: Processing "
             << reflection_angles.size() << " events..." << endl;

        TH1D* hist = createHistogram(reflection_angles, phi_target, bin_size, margin);

        FitResult result;
        result.phi = phi_target;
        result.n_events = reflection_angles.size();

        TF1* fit = performFit(hist, phi_target, result);
        results.push_back(result);

        cout << "φ = " << setw(2) << phi_target << "°: "
             << "p₁=" << fixed << setprecision(3) << result.p1
             << "±" << result.p1_err
             << ", p₂=" << result.p2 << "±" << result.p2_err
             << ", p₃=" << setprecision(1) << result.p3 << "±" << result.p3_err << "°"
             << ", p₄=" << result.p4 << "±" << result.p4_err << "°"
             << ", Ratio=" << result.ratio << "±" << result.ratio_err
             << ", χ²/ndf=" << result.chi2/result.ndof
             << ", n=" << result.n_events << endl;

        logfile << "φ = " << setw(2) << phi_target << "°: "
                << "p₁=" << fixed << setprecision(3) << result.p1
                << "±" << result.p1_err
                << ", p₂=" << result.p2 << "±" << result.p2_err
                << ", p₃=" << setprecision(1) << result.p3 << "±" << result.p3_err << "°"
                << ", p₄=" << result.p4 << "±" << result.p4_err << "°"
                << ", Ratio=" << result.ratio << "±" << result.ratio_err
                << ", χ²/ndf=" << result.chi2/result.ndof
                << ", n=" << result.n_events << endl;

        canvas->cd(idx + 1);
        gPad->SetGrid();
        gPad->SetTopMargin(0.12);
        gPad->SetRightMargin(0.05);

        hist->Draw("E");
        fit->SetLineColor(kRed);
        fit->SetLineWidth(2);
        fit->Draw("same");

        TLegend* legend = new TLegend(0.60, 0.70, 0.88, 0.88);
        legend->SetFillStyle(0);
        legend->SetBorderSize(0);
        legend->SetTextSize(0.045);
        legend->AddEntry(hist, "Data", "ep");
        legend->AddEntry(fit, "Gaussian + Lambertian fit", "l");
        legend->Draw();

        TPaveText* pt = new TPaveText(0.15, 0.78, 0.40, 0.88, "NDC");
        pt->SetFillStyle(0);
        pt->SetBorderSize(0);
        pt->SetTextSize(0.04);
        pt->AddText(Form("n = %d", result.n_events));
        pt->AddText(Form("#chi^{2}/ndf = %.2f", result.chi2/result.ndof));
        pt->Draw();

        delete hist;
    }

    // Save the main canvas (your original PDF output - keep as is)
    string output_pdf = output_dir + "/" + file_prefix + "_angular_analysis.pdf";
    string output_png = output_dir + "/" + file_prefix + "_angular_analysis.png";
    
    canvas->Print(output_pdf.c_str());
    canvas->Print(output_png.c_str());
    cout << "\n✓ Main plot saved as: " << output_pdf << endl;
    cout << "✓ Main plot saved as: " << output_png << endl;
    logfile << "\n✓ Main plot saved as: " << output_pdf << endl;

    // ============================================================================
    // NEW VALIDATION PLOTS (PNG only - does not remove your existing outputs)
    // ============================================================================
    
    // 2D Correlation Plot
    create2DCorrelationPlot(tree, output_dir, file_prefix);
    
    // Function A Validation (Interpolation between incident angles)
    validateFunctionA(tree, target_angles, output_dir, file_prefix);
    
    // Function B Validation (Continuous PDF interpolation)
    validateFunctionB(output_dir, file_prefix);
    
    // Function C Validation (Sampling reproduces PDFs)
    validateFunctionC(tree, target_angles, output_dir, file_prefix);

    // Your existing comparison plots (keep as is)
    if (!results.empty()) {
        printThesisTable(results, logfile);
        printComparisonTable(results, logfile);
        createComparisonPlot(results, output_dir, file_prefix);
    }

    cout << "\n" << string(60, '=') << endl;
    cout << "ANALYSIS COMPLETE" << endl;
    cout << string(60, '=') << endl;
    cout << "Input file: " << filename << endl;
    cout << "Output directory: " << output_dir << endl;
    cout << "Total reflections: " << nEntries << endl;
    cout << "Log file: " << log_filename << endl;
    cout << "\n--- VALIDATION OUTPUT FILES (PNG) ---" << endl;
    cout << "  - " << file_prefix << "_function_a_validation.png" << endl;
    cout << "  - " << file_prefix << "_function_b_validation.png" << endl;
    cout << "  - " << file_prefix << "_function_c_validation.png" << endl;
    cout << "  - " << file_prefix << "_2d_correlation.png" << endl;
    cout << string(60, '=') << endl;

    logfile << "\n" << string(60, '=') << endl;
    logfile << "ANALYSIS COMPLETE" << endl;
    logfile << string(60, '=') << endl;

    delete canvas;
    file->Close();
    logfile.close();
}

// ============================================================================
// Print usage information
// ============================================================================
void printUsage(const char* program_name) {
    cout << "\nUsage: " << program_name << " <input_file>" << endl;
    cout << "\nExample:" << endl;
    cout << "  " << program_name << " Sim_D2ODetector130000.root" << endl;
    cout << "  " << program_name << " /full/path/to/Sim_D2ODetector130000.root" << endl;
    cout << "\nOutput:" << endl;
    cout << "  - " << getBaseFilename("Sim_D2ODetector130000.root") << "_angular_analysis.pdf/png (original)" << endl;
    cout << "  - " << getBaseFilename("Sim_D2ODetector130000.root") << "_2d_correlation.png (new)" << endl;
    cout << "  - " << getBaseFilename("Sim_D2ODetector130000.root") << "_function_a_validation.png (new)" << endl;
    cout << "  - " << getBaseFilename("Sim_D2ODetector130000.root") << "_function_b_validation.png (new)" << endl;
    cout << "  - " << getBaseFilename("Sim_D2ODetector130000.root") << "_function_c_validation.png (new)" << endl;
    cout << endl;
}

// ============================================================================
// Main function
// ============================================================================
int main(int argc, char** argv) {
    gStyle->SetPalette(55);
    gStyle->SetLabelSize(0.04, "xyz");
    gStyle->SetTitleSize(0.05, "xyz");
    gStyle->SetTitleOffset(1.2, "x");
    gStyle->SetTitleOffset(1.5, "y");

    if (argc < 2) {
        cerr << "\nERROR: No input file specified!" << endl;
        printUsage(argv[0]);
        return 1;
    }

    string input_file = argv[1];
    string output_dir = getCurrentDirectory() + "/tyvek_analysis_results";

    // Create output directory
    if (!createDirectory(output_dir)) {
        cerr << "ERROR: Could not create output directory: " << output_dir << endl;
        return 1;
    }

    cout << "\n" << string(60, '=') << endl;
    cout << "Tyvek Reflectivity Analysis Tool" << endl;
    cout << "WITH Functions A, B, C Validation" << endl;
    cout << string(60, '=') << endl;
    cout << "Input file: " << input_file << endl;
    cout << "Output directory: " << output_dir << endl;

    // Process the file
    processFile(input_file, output_dir);

    cout << "\n" << string(60, '=') << endl;
    cout << "ANALYSIS COMPLETE!" << endl;
    cout << "Results saved in: " << output_dir << endl;
    cout << string(60, '=') << endl;

    return 0;
}
