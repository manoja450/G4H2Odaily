
#include "G4d2oEventAction.hh"
#include "G4d2oDetectorHit.hh"
#include "G4d2oNeutrinoAlley.hh"
#include "G4d2oSensitiveDetector.hh"
#include "G4d2oDetector.hh"
#include "G4d2oRunAction.hh"
#include "G4Navigator.hh"
#include "G4TransportationManager.hh"

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4UnitsTable.hh"
#include "G4RunManager.hh"

#include "TFile.h"
#include "TMath.h"
#include "TString.h"
#include "TClassTable.h"
#include "inputVariables.hh"

G4d2oEventAction::G4d2oEventAction(void)
{
    G4cerr << "\tConstructing G4d2oEventAction...";

    // Initialize pointers
    HCoE = NULL;
    for (G4int i = 0; i < MAX_SEN_DET; i++)
    {
        detHC[i] = NULL;
        sourceHC[i] = NULL;
        targHC[i] = NULL;
    }

    numEvents = 0;

    input = inputVariables::GetIVPointer();

    G4d2oNeutrinoAlley *detCon = (G4d2oNeutrinoAlley *)G4RunManager::GetRunManager()->GetUserDetectorConstruction();
    theDet = (G4d2oDetector *)detCon->GetDetectorPtr();

    G4d2oRunAction *runAct = (G4d2oRunAction *)G4RunManager::GetRunManager()->GetUserRunAction();
    fout = TFile::Open(runAct->GetOutFileName(), "recreate");

    if (!TClassTable::GetDict("simEvent"))
    {
        G4cout << "Error in G4d2oEventAction::G4d2oEventAction() - class simEvent not found\n"
               << G4endl;
        exit(0);
    }
    if (!TClassTable::GetDict("simHit"))
    {
        G4cout << "Error in G4d2oEventAction::G4d2oEventAction() - class simHit not found\n"
               << G4endl;
        exit(0);
    }
    if (!TClassTable::GetDict("simAreaHit"))
    {
        G4cout << "Error in G4d2oEventAction::G4d2oEventAction() - class simAreaHit not found\n"
               << G4endl;
        exit(0);
    }
    theEventData = new simEvent();

    outTree = new TTree("Sim_Tree", "tree");
    outTree->Branch("eventData", "simEvent", &theEventData);

    numPMTs = theDet->GetTotalPMTS();
    numRows = theDet->GetPMTRows();
    numInEachRow = theDet->GetPMTsInRow();

    if (numPMTs != numRows * numInEachRow)
    {
        printf("\nError in G4d2oEventAction::G4d2oEventAction() - %d x %d != %d for PMT array.\n\n", numRows, numInEachRow, numPMTs);
        exit(0);
    }

    // some histograms
    if (numPMTs > 0 && numRows > 0 && numInEachRow > 0)
    {
        hPMTArray = new TH2D("hPMTArray", "PMT hit pattern", numInEachRow, -0.5, numInEachRow - 0.5, numRows, -0.5, numRows - 0.5);
        hPMTNumVsTime = new TH2D("hPMTNumVsTime", "PMT number vs. time", 100, 0, 100, numPMTs, -0.5, numPMTs - 0.5);
    }
    else
    {
        hPMTArray = 0;
        hPMTNumVsTime = 0;
    }
    //    hPhotonEnergy = new TH1D("hPhotonEnergy","Photon Energy",500,0,1000.0);
    hPhotonEnergy = new TH1D("hPhotonEnergy", "Photon Energy", 500, 1.0, 10.0);
    hTotalPhotons = new TH1D("hTotalPhotons", "Total photons", 1000, 0, 10000);

    for (Int_t i = 0; i < 2; i++)
        hSidePMT[i] = 0;

    if (input->GetSideLining() == 1 && theDet->SidePMTX() > 0 && theDet->SidePMTZ() > 0)
    {
        for (Int_t i = 0; i < 2; i++)
        {
            hSidePMT[i] = new TH2D(Form("hSidePMT[%d]", i), Form("Side PMT %d", i),
                                   500, -theDet->SidePMTX() / 2.0, theDet->SidePMTX() / 2.0,
                                   500, -theDet->SidePMTZ() / 2.0, theDet->SidePMTZ() / 2.0);
        }
    }

    thePGA = (G4d2oPrimaryGeneratorAction *)G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction();
    /*
        // Photons are now thrown out at generation rather than detection
        const double hc_evnm = 1.23984193 *1e3;

        std::vector<double> qe_bialkali_y = {0.000,0.000,0.000,0.000,
          0.001,0.003,0.008,0.017,
          0.033,0.052,0.076,0.106,
          0.139,0.164,0.193,0.204,
          0.228,0.240,0.254,0.254,
          0.240,0.213,0.164,0.032,
          0.003,0.0};

        std::vector<double> qe_bialkali_xgev = {1.63e-9,1.68e-9,1.72e-9,1.77e-9,
          1.82e-9,1.88e-9,1.94e-9,2.00e-9,
          2.07e-9,2.14e-9,2.21e-9,2.30e-9,
          2.38e-9,2.48e-9,2.58e-9,2.70e-9,
          2.82e-9,2.95e-9,3.10e-9,3.26e-9,
          3.44e-9,3.65e-9,3.88e-9,4.13e-9,
          4.43e-9,4.7e-9};

        std::vector<double> qe_bialkali_xnm;

        for(auto xgev : qe_bialkali_xgev) qe_bialkali_xnm.push_back(hc_evnm/(xgev*1e9));

        gBialkali_ev = new TGraph(qe_bialkali_xgev.size());

        for(int ipoint = 0; ipoint<qe_bialkali_xnm.size(); ipoint++ ){
            gBialkali_ev->SetPoint(ipoint,qe_bialkali_xgev[ipoint]*1e9,qe_bialkali_y[ipoint]);
        }
    */

    G4cerr << "done." << G4endl;

} // END of constructor

G4d2oEventAction::~G4d2oEventAction()
{
    G4cout << "Deleting G4d2oEventAction";

    if (outTree)
    {
        outTree->AutoSave();

        fout->cd();
        fout->mkdir("histos");
        fout->cd("histos");

        if (hPMTArray)
            hPMTArray->Write("pmtHits");
        if (hPMTNumVsTime)
            hPMTNumVsTime->Write("pmtNumVsTime");
        if (hPhotonEnergy)
            hPhotonEnergy->Write("photonEnergy");
        if (hTotalPhotons)
            hTotalPhotons->Write("photonsPerEvent");

        if (input->GetSideLining() == 1)
        {
            for (Int_t i = 0; i < 2; i++)
                if (hSidePMT[i])
                    hSidePMT[i]->Write(Form("sidePMTHits%d", i));
        }

        fout->mkdir("pmtPositions");
        fout->cd("pmtPositions");
        // save histograms with coordinates of the front faces of PMTs
        if (theDet->GetPMTPositionHistogram(0))
            theDet->GetPMTPositionHistogram(0)->Write("pmtPosX");
        if (theDet->GetPMTPositionHistogram(1))
            theDet->GetPMTPositionHistogram(1)->Write("pmtPosY");
        if (theDet->GetPMTPositionHistogram(2))
            theDet->GetPMTPositionHistogram(2)->Write("pmtPosZ");
    }

    G4String theOutFileName = fout->GetName();
    system("mkdir -p scripts");
    ofstream lastFile("scripts/lastFileName.txt");
    lastFile << fout->GetName() << endl;
    lastFile.close();

    delete fout;

    delete gBialkali_ev;

    G4cout << "...done" << G4endl << G4endl;

    G4cout << "\t**********************************************************************************" << G4endl;
    G4cout << "\t***   Output File: " << theOutFileName << " " << G4endl;
    G4cout << "\t**********************************************************************************" << G4endl;
    G4cout << G4endl << G4endl;

} // END of constructor

void G4d2oEventAction::BeginOfEventAction(const G4Event *thisEvent)
{

    fVetoEdep = 0.0;

    ResetNarrowVetoEdep();
    ResetWideVetoEdep();
    //    thisEvent = 0;

} // END of BeginOfEventAction()

void G4d2oEventAction::EndOfEventAction(const G4Event *thisEvent)
{
    HCoE = thisEvent->GetHCofThisEvent();
    if (HCoE)
    {
        GetHitsCollection();
        ProcessEvent();
    }
    for (int i = 0; i < kNarrowPanels; ++i)
        theEventData->narrow_veto_edep[i] = fNarrowVetoEdep[i];
    for (int i = 0; i < kWidePanels; ++i)
        theEventData->wide_veto_edep[i] = fWideVetoEdep[i];
}
// END of EndOfEventAction()

void G4d2oEventAction::GetHitsCollection(void)
{
    for (Int_t i = 0; i < input->numDetHC; i++)
        detHC[i] = (G4d2oDetectorHitsCollection *)HCoE->GetHC(input->detCollID[i]);
    for (Int_t i = 0; i < input->numSrcHC; i++)
        sourceHC[i] = (G4d2oDetectorHitsCollection *)HCoE->GetHC(input->srcCollID[i]);
    for (Int_t i = 0; i < input->numTargHC; i++)
        targHC[i] = (G4d2oDetectorHitsCollection *)HCoE->GetHC(input->targCollID[i]);

} // END of GetHitsCollection()

void G4d2oEventAction::ProcessEvent(void)
{
    theEventData->ClearData();
    ZeroEventVariables();
    numEvents++;

    eventNumber = numEvents;

    G4d2oDetectorHit *thisHit;

    G4ThreeVector initDir = thePGA->GetInitialDirection();
    theEventData->direction0.SetXYZ(initDir.x(), initDir.y(), initDir.z());
    G4ThreeVector initPos = thePGA->GetOriginalPosition();
    theEventData->position0.SetXYZ(initPos.x(), initPos.y(), initPos.z());
    theEventData->sourceParticleEnergy = thePGA->GetSourceEnergy();
    theEventData->eventNumber = eventNumber;
    theEventData->vol0 = -1;

    G4Navigator *Navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();

    G4VPhysicalVolume *volume0 = Navigator->LocateGlobalPointAndSetup(initPos);
    if (volume0->GetName() == "h2oPhysV")
    {
        theEventData->vol0 = 2;
    }
    else if (volume0->GetName() == "d2oPhysV")
    {
        theEventData->vol0 = 1;
    }
    else if (volume0->GetName() == "acrylicPhysV")
    {
        theEventData->vol0 = 3;
    }
    else if (volume0->GetName() == "pmtPhysV_0")
    {
        theEventData->vol0 = 4;
    }
    else if (volume0->GetName() == "pmtPhysV_1")
    {
        theEventData->vol0 = 5;
    }
    else if (volume0->GetName() == "pmtPhysV_2")
    {
        theEventData->vol0 = 6;
    }
    else if (volume0->GetName() == "pmtPhysV_3")
    {
        theEventData->vol0 = 7;
    }
    else if (volume0->GetName() == "pmtPhysV_4")
    {
        theEventData->vol0 = 8;
    }
    else if (volume0->GetName() == "pmtPhysV_5")
    {
        theEventData->vol0 = 9;
    }
    else if (volume0->GetName() == "pmtPhysV_6")
    {
        theEventData->vol0 = 10;
    }
    else if (volume0->GetName() == "pmtPhysV_7")
    {
        theEventData->vol0 = 11;
    }
    else if (volume0->GetName() == "pmtPhysV_8")
    {
        theEventData->vol0 = 12;
    }
    else if (volume0->GetName() == "pmtPhysV_9")
    {
        theEventData->vol0 = 13;
    }
    else if (volume0->GetName() == "pmtPhysV_10")
    {
        theEventData->vol0 = 14;
    }
    else if (volume0->GetName() == "pmtPhysV_11")
    {
        theEventData->vol0 = 15;
    }
    else if (volume0->GetName() == "pmtPhysV_12")
    {
        theEventData->vol0 = 16;
    }
    else if (volume0->GetName() == "pmtPhysV_13")
    {
        theEventData->vol0 = 17;
    }
    else if (volume0->GetName() == "pmtPhysV_14")
    {
        theEventData->vol0 = 18;
    }
    else if (volume0->GetName() == "pmtPhysV_15")
    {
        theEventData->vol0 = 19;
    }
    else if (volume0->GetName() == "pmtPhysV_16")
    {
        theEventData->vol0 = 20;
    }
    else if (volume0->GetName() == "pmtPhysV_17")
    {
        theEventData->vol0 = 21;
    }
    else if (volume0->GetName() == "pmtPhysV_18")
    {
        theEventData->vol0 = 22;
    }
    else if (volume0->GetName() == "pmtPhysV_19")
    {
        theEventData->vol0 = 23;
    }
    else if (volume0->GetName() == "pmtPhysV_20")
    {
        theEventData->vol0 = 24;
    }
    else if (volume0->GetName() == "pmtPhysV_21")
    {
        theEventData->vol0 = 25;
    }
    else if (volume0->GetName() == "pmtPhysV_22")
    {
        theEventData->vol0 = 26;
    }
    else if (volume0->GetName() == "pmtPhysV_23")
    {
        theEventData->vol0 = 27;
    }
    else if (volume0->GetName() == "pmtPhysV_24")
    {
        theEventData->vol0 = 28;
    }
    else if (volume0->GetName() == "pmtPhysV_25")
    {
        theEventData->vol0 = 29;
    }
    else if (volume0->GetName() == "pmtPhysV_26")
    {
        theEventData->vol0 = 30;
    }
    else if (volume0->GetName() == "pmtPhysV_27")
    {
        theEventData->vol0 = 31;
    }
    else if (volume0->GetName() == "pmtPhysV_28")
    {
        theEventData->vol0 = 32;
    }
    else if (volume0->GetName() == "pmtPhysV_29")
    {
        theEventData->vol0 = 33;
    }
    else if (volume0->GetName() == "pmtPhysV_30")
    {
        theEventData->vol0 = 34;
    }
    else if (volume0->GetName() == "pmtPhysV_31")
    {
        theEventData->vol0 = 35;
    }
    else if (volume0->GetName() == "pmtPhysV_32")
    {
        theEventData->vol0 = 36;
    }
    else if (volume0->GetName() == "pmtPhysV_33")
    {
        theEventData->vol0 = 37;
    }
    else if (volume0->GetName() == "pmtPhysV_34")
    {
        theEventData->vol0 = 38;
    }
    else if (volume0->GetName() == "pmtPhysV_35")
    {
        theEventData->vol0 = 39;
    }
    else if (volume0->GetName() == "pmtPhysV_36")
    {
        theEventData->vol0 = 40;
    }
    else if (volume0->GetName() == "pmtPhysV_37")
    {
        theEventData->vol0 = 41;
    }
    else if (volume0->GetName() == "pmtPhysV_38")
    {
        theEventData->vol0 = 42;
    }
    else if (volume0->GetName() == "pmtPhysV_39")
    {
        theEventData->vol0 = 43;
    }
    else if (volume0->GetName() == "pmtPhysV_40")
    {
        theEventData->vol0 = 44;
    }
    else if (volume0->GetName() == "pmtPhysV_41")
    {
        theEventData->vol0 = 45;
    }
    else if (volume0->GetName() == "pmtPhysV_42")
    {
        theEventData->vol0 = 46;
    }
    else if (volume0->GetName() == "pmtPhysV_43")
    {
        theEventData->vol0 = 47;
    }
    else if (volume0->GetName() == "pmtPhysV_44")
    {
        theEventData->vol0 = 48;
    }
    else if (volume0->GetName() == "pmtPhysV_45")
    {
        theEventData->vol0 = 49;
    }
    else if (volume0->GetName() == "pmtPhysV_46")
    {
        theEventData->vol0 = 50;
    }
    else if (volume0->GetName() == "pmtPhysV_47")
    {
        theEventData->vol0 = 51;
    }

    // Form("pmtPhysV_%d",iCopy)

    // TODO: Ask Matthew where I can put this more globally
    std::map<std::string, int> vmuid = {{"muVetoBI", 0}, {"muVetoBO", 1}, {"muVetoTI", 2}, {"muVetoTO", 3}, {"muVetoLI", 4}, {"muVetoLO", 5}, {"muVetoRI", 6}, {"muVetoRO", 7}, {"muVetoNI", 8}, {"muVetoNO", 9}, // Near to walkway
                                        {"muVetoFI", 10},
                                        {"muVetoFO", 11}}; // Far to walkway, against wall

    // Loop through the detector hits collection
    for (G4int iDetHC = 0; iDetHC < input->numDetHC; iDetHC++)
    {
        if (detHC[iDetHC])
        {
            TString SDname = detHC[iDetHC]->GetSDname().data();
            if (SDname.BeginsWith("muVetoInner"))
            {
                int muid = 0;
                for (G4int i = 0; i < detHC[iDetHC]->entries(); i++)
                {
                    thisHit = (*detHC[iDetHC])[i];
                    if (thisHit->GetPDGCharge() != 0)
                    {
                        theEventData->muVetoEnergy[muid] += thisHit->GetTotalEnergyDeposit();
                    }
                }
            }
            else if (SDname.BeginsWith("muVetoOuter"))
            {
                int muid = 1;
                for (G4int i = 0; i < detHC[iDetHC]->entries(); i++)
                {
                    thisHit = (*detHC[iDetHC])[i];
                    if (thisHit->GetPDGCharge() != 0)
                    {
                        theEventData->muVetoEnergy[muid] += thisHit->GetTotalEnergyDeposit();
                    }
                }
            }
            else if (SDname.BeginsWith("muVeto"))
            {
                int muid = vmuid[SDname.Data()];
                for (G4int i = 0; i < detHC[iDetHC]->entries(); i++)
                {
                    thisHit = (*detHC[iDetHC])[i];
                    if (thisHit->GetPDGCharge() != 0)
                    {
                        theEventData->muVetoEnergy[muid] += thisHit->GetTotalEnergyDeposit();
                    }
                }
            }
            else
            {
                for (G4int i = 0; i < detHC[iDetHC]->entries(); i++)
                {
                    thisHit = (*detHC[iDetHC])[i];

                    G4String hitParticleName = thisHit->GetParticleName();
                    if (hitParticleName == "opticalphoton")
                    {

                        G4int hitPMTNumber = thisHit->GetDetectorNumber();
                        G4double hitGlobalTime = thisHit->GetGlobalTime();
                        G4double hitKineticEnergy = thisHit->GetKineticEnergy();

                        // Photons now thrown out at generation not detection
                        // if(G4UniformRand()>qe_ev(hitKineticEnergy/eV)) continue;

                        //-----------------------------------------------------------

                        // 02/14/2023:
                        // NEW: Detecting everything based on individual PMT QE,
                        //     obtained from DATA and MC p.e. distribution ratios.

#if 1
                        if (hitPMTNumber == 0)
                        { // pmt_qe_scaling = 0.8329
                            if (G4UniformRand() > 0.8329)
                                continue;
                        }

                        if (hitPMTNumber == 1)
                        { // pmt_qe_scaling=0.8197
                            if (G4UniformRand() > 0.8197)
                                continue;
                        }

                        if (hitPMTNumber == 2)
                        { // pmt_qe_scaling=0.9205
                            if (G4UniformRand() > 0.9205)
                                continue;
                        }

                        if (hitPMTNumber == 3)
                        { // pmt_qe_scaling=0.6747
                            if (G4UniformRand() > 0.6747)
                                continue;
                        }

                        if (hitPMTNumber == 4)
                        { // pmt_qe_scaling=0.8538
                            if (G4UniformRand() > 0.8538)
                                continue;
                        }

                        if (hitPMTNumber == 5)
                        { // BAD PMT
                            if (G4UniformRand() > 1.0000)
                                continue;
                        }

                        if (hitPMTNumber == 6)
                        { // pmt_qe_scaling=0.9951
                            if (G4UniformRand() > 0.9951)
                                continue;
                        }

                        if (hitPMTNumber == 7)
                        { // pmt_qe_scaling=0.8934
                            if (G4UniformRand() > 0.8934)
                                continue;
                        }

                        if (hitPMTNumber == 8)
                        { // pmt_qe_scaling=0.7615
                            if (G4UniformRand() > 0.7615)
                                continue;
                        }

                        if (hitPMTNumber == 9)
                        { // BAD PMT
                            if (G4UniformRand() > 1.0000)
                                continue;
                        }

                        if (hitPMTNumber == 10)
                        { // pmt_qe_scaling=0.6164
                            if (G4UniformRand() > 0.6164)
                                continue;
                        }

                        if (hitPMTNumber == 11)
                        { // The one that sees more light
                            if (G4UniformRand() > 1.0000)
                                continue;
                        }
#endif

                        //-----------------------------------------------------------

                        // normal PMTs
                        if (detHC[iDetHC]->GetSDname() == "pmt")
                        {
                            theEventData->AddPMTHit(hitPMTNumber, hitGlobalTime, hitKineticEnergy);
                            if (hPMTArray)
                                hPMTArray->Fill(hitPMTNumber / numRows, hitPMTNumber % numRows);
                            if (hPMTNumVsTime)
                                hPMTNumVsTime->Fill(hitGlobalTime, hitPMTNumber);
                        }
#if 0
                        else{
                            TVector3 hitPosition(thisHit->GetPosition().x(),thisHit->GetPosition().y(),thisHit->GetPosition().z());
                            theEventData->AddAreaPMTHit(hitPMTNumber, hitGlobalTime, hitKineticEnergy, hitPosition);
                            if(hitPMTNumber>=0 && hitPMTNumber<=1){
                                if(hSidePMT[hitPMTNumber]) hSidePMT[hitPMTNumber]->Fill(hitPosition.X(),hitPosition.Z());
                            }
                        }
#endif
                        hPhotonEnergy->Fill(hitKineticEnergy / eV);

                    } // check for an opticalphoton

                } // loop over hits (i)
            } // if no muon Veto Hits Collection

            if (detHC[iDetHC]->GetSDname() == "topVetoSD") // && detHC[iDetHC]->GetSize() > 0)
            {
                if (detHC[iDetHC]->GetSize() > 0)
                {
                    theEventData->veto_tag = true;
                    theEventData->veto_edep = fVetoEdep;
                }
                else
                {
                    theEventData->veto_tag = false;
                    theEventData->veto_edep = fVetoEdep;
                }
            }
            //G4cout << "[EA] SD name seen: " << detHC[iDetHC]->GetSDname() << G4endl;
if (detHC[iDetHC]->GetSDname() == "VetoNu1SD") {
  for (int i=0;i<kNarrowPanels;++i)
    theEventData->narrow_veto_edep[i] = fNarrowVetoEdep[i];
}

if (detHC[iDetHC]->GetSDname() == "VetoNu2SD") {
  for (int i=0;i<kWidePanels;++i)
    theEventData->wide_veto_edep[i] = fWideVetoEdep[i];
}

        } // detHC exists
    } // loop over dethits collections
    if (eventNumber % 1000 == 0)
    {
        G4cout << "Completed Event " << eventNumber << G4endl;
        if (input->GetPGAType() == 10)
        {
            G4cout << "Special Energy = " << input->GetSpecialEnergy() << G4endl;
        }
    }

    // Save to detector variables to pass to tree
    // if(theEventData->numHits>0)
    //outTree->Branch("narrow_veto_edep", theEventData->narrow_veto_edep, "narrow_veto_edep[4]/D");
    outTree->Fill();

    hTotalPhotons->Fill(theEventData->numHits);

} // END of ProcessEvent()

void G4d2oEventAction::ZeroEventVariables(void)
{

    eventNumber = 0;

} // END of ZeroEventVariables()

double G4d2oEventAction::qe_ev(double energy_ev)
{
    if (!gBialkali_ev)
        return 0.0;
    if (energy_ev < gBialkali_ev->GetX()[0] || energy_ev > gBialkali_ev->GetX()[gBialkali_ev->GetN() - 1])
        return 0.0;
    return gBialkali_ev->Eval(energy_ev, 0, "");
}
