
#include "G4d2oPhysicsList.hh"

#include <fstream>

G4d2oPhysicsList::G4d2oPhysicsList(G4bool bOn, G4bool bNHP)
: G4VModularPhysicsList()
{
    //the default cut value will only simulate things that extend
    //beyond this value.  Otherwise will treat as energy loss
    defaultCutValue = 0.01*mm;
    
    bNull = !(bOn);
    bNeutronHP = bNHP;
    
    if(bNull) G4cerr<<"\tG4d2oPhysicsList constructor: Physics list is empty. \n\t\tPass the argument 'bOn=true' to turn it on."<<G4endl;
    else{
        if(!bNeutronHP) G4cerr<<"\tG4d2oPhysicsList constructor: Using non-HP neutron physics. \n\t\tPass the argument 'bNHP=true' to turn on HP neutron physics."<<G4endl;
    }
    
    verbosityLevel = 1;
    //Verbosity level:
    //  0: Silent
    //  1: Warning message
    //  2: More
    SetVerboseLevel(verbosityLevel);
    
}

//destructor
G4d2oPhysicsList::~G4d2oPhysicsList() {
	G4cerr<<"Deleting G4d2oPhysicsList...";
	
	G4cerr<<"done"<<G4endl;
}

//---------------------------------------------------------
//Construct Particles 
//---------------------------------------------------------

//these methods construct all particles
#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"

//Construct the different types of particles with different functions.
//We construct all particles since this should not affect performance.
//We will not, however, add all the physics processes for each particle
//as this would affect performance.
//Furthermore, since particles are defined statically, they are
//all added to the G4ParticleTable regardless of what goes on in this method.
void G4d2oPhysicsList::ConstructParticle()
{
    
    G4cerr<<"\tConstructing particles...\n";
    
    //Construct all bosons 
    G4cerr<<"\t\tbosons"<<G4endl;
    G4BosonConstructor bConstructor;
    bConstructor.ConstructParticle();  
    
    //Construct all leptons
    G4cerr<<"\t\tleptons"<<G4endl;
    G4LeptonConstructor lConstructor;
    lConstructor.ConstructParticle();
    
    //Construct all mesons
    G4cerr<<"\t\tmesons"<<G4endl;
    G4MesonConstructor mConstructor;
    mConstructor.ConstructParticle();
    
    //Construct all baryons
    G4cerr<<"\t\tbaryons"<<G4endl;
    G4BaryonConstructor  hConstructor;
    hConstructor.ConstructParticle();  
    
    //Construct light ions
    G4cerr<<"\t\tlight ions"<<G4endl;
    G4IonConstructor iConstructor;
    iConstructor.ConstructParticle();  
    
    //Construct short lived as they are needed
    //for the electronuclear process
    G4cerr<<"\t\tshort lived"<<G4endl;
    G4ShortLivedConstructor sConstructor;
    sConstructor.ConstructParticle();

    G4cerr<<"\tDone constructing particles."<<G4endl;
    
}

//---------------------------------------------------------
//Construct Processes 
//---------------------------------------------------------


//needed to add and manage processes
#include "G4ProcessManager.hh"

void G4d2oPhysicsList::ConstructProcess()
{
    //adds transportation so particles can move
    //protected method of G4VUserPhysicsList
    AddTransportation();
    
    if(!bNull){
        //Electric and Magnetic Processes
        G4cerr << "G4d2oPhysicsList::ConstructProcess() -> Registering EM processes" << G4endl;;
        ConstructEM();
        
        //Radioactive Decay
        G4bool bTurnOnRAD = true;
        
        if(bTurnOnRAD){
            G4cerr << "G4d2oPhysicsList::ConstructProcess() -> Registering radioactive decay processes" << G4endl;
            ConstructRadioactiveDecay();
        }
        else{
            G4cerr << G4endl << "************************* WARNING ******************************" << G4endl;
            G4cerr << "G4d2oPhysicsList::ConstructProcess() -> NOT registering radioactive decay processes" << G4endl;;
            G4cerr << "************************* WARNING ******************************" << G4endl << G4endl;
            system("sleep 5.0");
        }
        
        //Construct hadronic physics
        G4cerr << "G4d2oPhysicsList::ConstructProcess() -> Registering hadronic processes" << G4endl;
        ConstructHadronic();
        
        //Construct optical physics
        G4cerr << "G4d2oPhysicsList::ConstructProcess() -> Registering optical processes" << G4endl;
        ConstructOptical();
    }
    
    G4cerr << "\tG4d2oPhysicsList::ConstructProcess() -> Done registering physics processes" << G4endl;
    
    
}

#include "G4EmLivermorePhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmExtraPhysics.hh"

//add all the EM processes
void G4d2oPhysicsList::ConstructEM()
{
    G4VPhysicsConstructor *emPhysicsList = new G4EmStandardPhysics(verbosityLevel);
    emPhysicsList->ConstructProcess();
    
    //synchrotron radiation, electro- and photo-nuclear
    G4VPhysicsConstructor *emExtraPhysicsList = new G4EmExtraPhysics(verbosityLevel);
    emExtraPhysicsList->ConstructProcess();
    
}//end of ConstructEM()

#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"

//add radioactive decay processes
void G4d2oPhysicsList::ConstructRadioactiveDecay()
{
    
    G4VPhysicsConstructor *decPhysicsList = new G4DecayPhysics();
    decPhysicsList->ConstructProcess();

    G4VPhysicsConstructor *radDecPhysicsList = new G4RadioactiveDecayPhysics();
    radDecPhysicsList->ConstructProcess();

}//end of ConstructRadioactiveDecay()

#include "G4HadronElasticPhysicsHP.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4HadronPhysicsQGSP_BIC_HP.hh"
#include "G4HadronPhysicsQGSP_BIC.hh"
#include "G4HadronPhysicsQGSP_BERT_HP.hh"
#include "G4HadronPhysicsQGSP_BERT.hh"
#include "G4StoppingPhysics.hh"
#include "G4IonBinaryCascadePhysics.hh"
#include "G4IonPhysics.hh"

//the hadronic interactions for neutrons, protons and ions
void G4d2oPhysicsList::ConstructHadronic()
{

    //all physics below is modeled on either QGSP_BIC(_HP) or QGSP_BERT(_HP) physics lists
    G4bool bBIC = true;

    G4VPhysicsConstructor *hadStopping = 0;
    G4VPhysicsConstructor *hadElastic = 0;
    G4VPhysicsConstructor *hadPhysics = 0;
    G4VPhysicsConstructor *ionPhysics = 0;
    
    // Hadron Elastic G4d2oing
    if(bNeutronHP) hadElastic = new G4HadronElasticPhysicsHP(verbosityLevel);
    else hadElastic = new G4HadronElasticPhysics(verbosityLevel);
        
    // Ion and Hadron physics
    if(bBIC){
        ionPhysics = new G4IonBinaryCascadePhysics(verbosityLevel);  //from BIC
        if(bNeutronHP) hadPhysics = new G4HadronPhysicsQGSP_BIC_HP(verbosityLevel);
        else hadPhysics = new G4HadronPhysicsQGSP_BIC(verbosityLevel);
    }
    else{
        ionPhysics = new G4IonPhysics(verbosityLevel);                   //from BERT
        if(bNeutronHP) hadPhysics = new G4HadronPhysicsQGSP_BERT_HP(verbosityLevel);
        else hadPhysics = new G4HadronPhysicsQGSP_BERT(verbosityLevel);        
    }

    // Stopping Physics
    hadStopping = new G4StoppingPhysics(verbosityLevel);

    if(hadElastic) hadElastic->ConstructProcess();
    if(hadPhysics) hadPhysics->ConstructProcess();
    if(hadStopping) hadStopping->ConstructProcess();
    if(ionPhysics) ionPhysics->ConstructProcess();
    
}//end of ConstructHadronic()

#include "G4Scintillation.hh"
#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4Cerenkov.hh"

void G4d2oPhysicsList::ConstructOptical()
{
    
  G4Scintillation* theScintillationProcess = new G4Scintillation("Scintillation");
  //  theScintillationProcess->SetScintillationYieldFactor(1.);
  theScintillationProcess->SetTrackSecondariesFirst(true);

    //G4Cerenkov treatment taken from G4 Application Developers guide
    G4int MaxNumPhotons = 300;
    G4Cerenkov *theCerenkovProcess = new G4Cerenkov("Cerenkov");
    theCerenkovProcess->SetTrackSecondariesFirst(true);
    theCerenkovProcess->SetMaxBetaChangePerStep(10.0);
    theCerenkovProcess->SetMaxNumPhotonsPerStep(MaxNumPhotons);

    G4OpAbsorption* theAbsorptionProcess = new G4OpAbsorption();
    G4OpRayleigh* theRayleighScatteringProcess = new G4OpRayleigh();
    G4OpBoundaryProcess* theBoundaryProcess = new G4OpBoundaryProcess();
    
    GetParticleIterator()->reset();
    
    while( (*GetParticleIterator())() ){
        G4ParticleDefinition* particle = GetParticleIterator()->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();
        if (theScintillationProcess->IsApplicable(*particle)) {
            pmanager->AddProcess(theScintillationProcess);
            pmanager->SetProcessOrderingToLast(theScintillationProcess, idxAtRest);
            pmanager->SetProcessOrderingToLast(theScintillationProcess, idxPostStep);
        }
        if (theCerenkovProcess->IsApplicable(*particle)) {
            pmanager->AddProcess(theCerenkovProcess);
            pmanager->SetProcessOrdering(theCerenkovProcess,idxPostStep);
        }
        if (particleName == "opticalphoton") {
            G4cout << " AddDiscreteProcess to OpticalPhoton " << G4endl;
            pmanager->AddDiscreteProcess(theAbsorptionProcess);
            pmanager->AddDiscreteProcess(theRayleighScatteringProcess);
            pmanager->AddDiscreteProcess(theBoundaryProcess);
        }
    }
    
}

//---------------------------------------------------------
// Set Cuts 
//---------------------------------------------------------

#include "G4UnitsTable.hh"

void G4d2oPhysicsList::SetCuts()
{
    if (verboseLevel >0)
    {
        G4cerr << "\tG4d2oPhysicsList::SetCuts:";
        G4cerr << "\tCutLength : " << G4BestUnit(defaultCutValue,"Length") << G4endl;
    }
    
    // set cut values for gamma at first and for e- second and next for e+,
    // because some processes for e+/e- need cut values for gamma
    SetCutValue(defaultCutValue, "gamma");
    SetCutValue(defaultCutValue, "e-");
    SetCutValue(defaultCutValue, "e+");
    
    if (verboseLevel>0) DumpCutValuesTable();
    G4cerr<<"\tDone initializing G4d2oPhysicsList"<<G4endl;
}


//end of file G4d2oPhysicsList.cc 
