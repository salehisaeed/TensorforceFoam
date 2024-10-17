/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015 OpenFOAM Foundation
    Copyright (C) 2016-2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "agentJetFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "probes.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::agentJetFvPatchVectorField::writeStateAction
(
    const scalarField& state,
    const scalar actionNew
)
{
    Ostream& os = file();
    writeCurrentTime(os);

    os  << actionNew;
    forAll(state, i)
    {
        os  << tab << state[i];
    }    
    os  << endl;
}

void Foam::agentJetFvPatchVectorField::writeFileHeader(Ostream& os)
{
    writeHeader(os, "Trajectory actions and states");
    writeCommented(os, "Time");
    writeCommented(os, "Action");
    writeCommented(os, "State (" + Foam::name(stateProbeLocations_.size()) + ")");
    os << endl;
}

Foam::scalarField Foam::agentJetFvPatchVectorField::environmentState()
{
    dictionary probesDict;
    probesDict.add("type", probes::typeName);
    probesDict.add("fixedLocations", "true");
    probesDict.add("fields", "()"); // The field should defined in the BC dictionary
    probesDict.add("interpolationScheme", interpolationScheme_);
    probesDict.add("probeLocations", stateProbeLocations_);
    probes p
    (
        "probes",
        internalField().mesh().time(),
        probesDict
    );
    
    return p.sample<scalar>(stateFieldName_);
}


Foam::scalar Foam::agentJetFvPatchVectorField::agentAction(const scalarField& state)
{
    //convert scalarField to std::vector<float>, TODO: find a smarter way!
    std::vector<float> stateVec(state.size());
    forAll(state, i)
    {
        stateVec[i] = state[i];
    }

    // Creating the input tensors of the policy model
    cppflow::tensor stateTensor(stateVec, {1, state.size()});
    std::vector<bool> det(1, deterministic_);
    cppflow::tensor detTensor(det, {});

    // Feeding the inputs to the loaded model to create an output (action)
    // The string arguements are found using saved_model_cli of Tensorflow
    auto action = model_
    (
        {
            {"serving_default_args_0:0", stateTensor}, // Model input 0,
            {"serving_default_deterministic:0", detTensor} // Model input 1
        },
        {
            "StatefulPartitionedCall:0" // Model output
        }
    );

    return action[0].get_data<float>()[0];
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF),
    functionObjects::writeFile(db(), typeName, "ActionState"),
    deterministic_(false),
    controlPeriod_(0),
    rampUpPeriod_(0),
    actionNew_(0),
    actionOld_(0),
    curTimeIndex_(-1),
    stateFieldName_(),
    stateProbesNo_(),
    stateProbeLocations_(Zero),
    interpolationScheme_(),
    policyDirName_(),
    model_(cppflow::model("model"))
{}


Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const agentJetFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
    functionObjects::writeFile(ptf),    
    deterministic_(ptf.deterministic_),
    controlPeriod_(ptf.controlPeriod_),
    rampUpPeriod_(ptf.rampUpPeriod_),
    actionNew_(ptf.actionNew_),
    actionOld_(ptf.actionOld_),
    curTimeIndex_(ptf.curTimeIndex_),
    stateFieldName_(ptf.stateFieldName_),
    stateProbesNo_(ptf.stateProbesNo_),
    stateProbeLocations_(ptf.stateProbeLocations_),
    interpolationScheme_(ptf.interpolationScheme_),
    policyDirName_(ptf.policyDirName_),
    model_(ptf.model_)
{}

Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict, false),
    functionObjects::writeFile(db(),typeName,"ActionState",dict),
    deterministic_(dict.get<bool>("deterministic")),
    controlPeriod_(dict.get<scalar>("controlPeriod")),
    rampUpPeriod_(dict.get<scalar>("rampUpPeriod")),
    actionNew_(0.0),
    actionOld_(dict.getOrDefault<scalar>("actionOld", 0.0)),
    curTimeIndex_(-1),
    stateFieldName_(dict.get<word>("stateField")),
    stateProbesNo_(dict.get<label>("stateProbesNo")),
    interpolationScheme_(dict.get<word>("interpolationScheme")),
    policyDirName_(dict.get<fileName>("policyDir")),
    model_
    (
        cppflow::model
        (
            fileName
            (
                db().time().globalPath()/policyDirName_
            )
        )
    )
{
    stateProbeLocations_ = vectorField
    (
        "stateProbeLocations",
        dict,
        stateProbesNo_
    );
    if (controlPeriod_ < rampUpPeriod_)
    {
        FatalErrorInFunction
            << "rampUpPeriod must be less that or equal to controlPeriod"
            << abort(FatalError);
    }
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=
        (
            vectorField("value", dict, p.size())
        );
    }
    else
    {
        updateCoeffs();
    }
    
    writeFileHeader(file());
}


Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const agentJetFvPatchVectorField& ptf
)
:
    fixedValueFvPatchField<vector>(ptf),
    functionObjects::writeFile(ptf),
    deterministic_(ptf.deterministic_),
    controlPeriod_(ptf.controlPeriod_),
    rampUpPeriod_(ptf.rampUpPeriod_),
    actionNew_(ptf.actionNew_),
    actionOld_(ptf.actionOld_),
    curTimeIndex_(ptf.curTimeIndex_),
    stateFieldName_(ptf.stateFieldName_),
    stateProbesNo_(ptf.stateProbesNo_),
    stateProbeLocations_(ptf.stateProbeLocations_),
    interpolationScheme_(ptf.interpolationScheme_),
    policyDirName_(ptf.policyDirName_),
    model_(ptf.model_)
{}


Foam::agentJetFvPatchVectorField::
agentJetFvPatchVectorField
(
    const agentJetFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF),
    functionObjects::writeFile(ptf),
    deterministic_(ptf.deterministic_),
    controlPeriod_(ptf.controlPeriod_),
    rampUpPeriod_(ptf.rampUpPeriod_),
    actionNew_(ptf.actionNew_),
    actionOld_(ptf.actionOld_),
    curTimeIndex_(ptf.curTimeIndex_),
    stateFieldName_(ptf.stateFieldName_),
    stateProbesNo_(ptf.stateProbesNo_),
    stateProbeLocations_(ptf.stateProbeLocations_),
    interpolationScheme_(ptf.interpolationScheme_),
    policyDirName_(ptf.policyDirName_),
    model_(ptf.model_) 
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::agentJetFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // Due to the inherent randomness of the agent's neural network, all the policy computations
    // are performed only once at each time step (e.g., the first outer corrector loop of PIMPLE)
    const label timeIndex = db().time().timeIndex();
    if (curTimeIndex_ != timeIndex)
    {
        const Time& time = db().time();
        scalar dt = time.deltaTValue();

        //TODO: What if period is not divisable by dt?
        const label nControlSteps = controlPeriod_ / dt;
        const label nRampSteps = rampUpPeriod_ / dt;

        scalar currentAction(actionNew_);
        label currentControlStep = (timeIndex % nControlSteps + nControlSteps) % nControlSteps;
        if (currentControlStep == 1)
        {
            Info<< "Updating agent action with policy model" << endl;
            scalarField state = environmentState();

            // Agian, due to randomness of the model, the new action is computed only for the
            // master processor and broadcast to other processors
            if (Pstream::master())
            {
                actionOld_ = actionNew_;
                actionNew_ = agentAction(state);
                writeStateAction(state, actionNew_);
                Info<< "New action = " 
                    << actionNew_
                    << ", Old action = "
                    << actionOld_ 
                    << endl;
            }
            // Broadcast the same action value on all processors when parallel processing
            Pstream::broadcast(actionNew_);
            Pstream::broadcast(actionOld_);
        }
        
        // Ramp up/down the new controller value
        scalar rampCoeff = 1;
        if ((currentControlStep <= nRampSteps) && (currentControlStep != 0))
        {
            rampCoeff = scalar(currentControlStep)/scalar(nRampSteps);
        }
        currentAction = rampCoeff*actionNew_ + (1 - rampCoeff)*actionOld_;

        vectorField::operator = (vector(0,currentAction,0));

        curTimeIndex_ = db().time().timeIndex();
    }

    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::agentJetFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry<bool>("deterministic", deterministic_);
    os.writeEntry("controlPeriod", controlPeriod_);
    os.writeEntry("rampUpPeriod", rampUpPeriod_);
    os.writeEntry<word>("policyDir", policyDirName_);
    os.writeEntry("actionNew", actionNew_);
    os.writeEntry("actionOld", actionOld_);
    os.writeEntry<word>("stateField", stateFieldName_);
    os.writeEntry("stateProbesNo", stateProbesNo_);
    stateProbeLocations_.writeEntry("stateProbeLocations", os);
    os.writeEntry<word>("interpolationScheme", interpolationScheme_);
    writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        agentJetFvPatchVectorField
    );
}

// ************************************************************************* //
