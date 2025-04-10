/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 Fredrik Dubay Myklebust, openfoam2006
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

Application
    calculateBdelta

Description
    Calculates the Reynolds stress anisotropy tensor bij and k deficit R
    using a frozen turbulent kinetic energy k field. The turbulence model is used
    to compute the required fields.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "/home/fredrik/OpenFOAM/fredrik-v2006/src/MLTurbulenceModels/frozensolver/frozensolver.H"




// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // OF headers
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createDyMControls.H"

    

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    // return set of times based on arglist
    const instantList& timeDirs = timeSelector::select0(runTime, args);

    forAll(timeDirs, timeI)
    {
        
        // Skip the first index
        if (timeI == 0) {
            continue;
        }
        // need to skip the second index as well for k_DNS_.oldTime().oldTime() to be used in ddt calc, a bit hacky approach
        if (timeI == 1) {
            continue;
        }
        

        // Set the time to the value of current time directory.
        runTime.setTime(timeDirs[timeI], timeI);
        Info<< "Time = " << runTime.timeName() << endl;
        Info<< "timeDirs[timeI] = " << timeDirs[timeI] << endl;

        Info<< "Reading the velocity field Udns\n" << endl;
        volVectorField U
        (
            IOobject
            (
                "Udns",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        Info<< "Reading the velocity field U\n" << endl;
        volVectorField Urans
        (
            IOobject
            (
                "Urans",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        Info<< "Reading nut \n" << endl;
        volScalarField nut
        (
            IOobject
            (
                "nut",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        Info<< "Reading the Reynolds stress tensor Rdns\n" << endl;
        volSymmTensorField Rdns
        (
            IOobject
            (
                "Rdns",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        Info<< "Reading the Reynolds stress tensor R\n" << endl;
        volSymmTensorField Rrans
        (
            IOobject
            (
                "Rrans",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );


        Info<< "Reading the turbulent kinetic energy kdns\n" << endl;
        volScalarField kdns
        (
            IOobject
            (
                "kdns",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        Info<< "Reading the turbulent kinetic energy k\n" << endl;
        volScalarField krans
        (
            IOobject
            (
                "krans",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );

        Info<< "Reading the turbulent omega\n" << endl;
        volScalarField omega
        (
            IOobject
            (
                "omega",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        Info<< "Reading field p\n" << endl;
        volScalarField p
        (
            IOobject
            (
                "p",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );

        #include "createPhi.H"

        label pRefCell = 0;
        scalar pRefValue = 0.0;
        setRefCell(p, pimple.dict(), pRefCell, pRefValue);
        mesh.setFluxRequired(p.name());


        singlePhaseTransportModel laminarTransport(U, phi);

        autoPtr<incompressible::turbulenceModel> turbulence
        (
            incompressible::turbulenceModel::New(U, phi, laminarTransport)
        );

        turbulence->validate();

        while (runTime.run())
        {
            Info<< "setting stoptime\n" << endl;
            Info<< "Time = " << runTime.timeName() << endl;
            Info<< "Solving the omega equation\n" << endl;
            laminarTransport.correct();
            turbulence->correct();
            ++runTime;
        }


        // The fields calculated by the turbulence model can be accessed with the 
        // lookupObject function. The fields are stored in the mesh object. 
        // not sure if this is the best way to access the fields, but works for now
        runTime.setTime(timeDirs[timeI], timeI);
        omega = turbulence->omega();
        omega.write();
        volScalarField kDeficit = mesh.lookupObject<volScalarField>("kDeficit");
        kDeficit.write();
        volSymmTensorField bijDelta = mesh.lookupObject<volSymmTensorField>("bijDelta");
        bijDelta.write();
        volSymmTensorField aijBoussinesq = mesh.lookupObject<volSymmTensorField>("aijBoussinesq");
        aijBoussinesq.write();
        volSymmTensorField aij_DNS = mesh.lookupObject<volSymmTensorField>("aij_DNS");
        aij_DNS.write();
        volSymmTensorField aijDelta = mesh.lookupObject<volSymmTensorField>("aijDelta");
        aijDelta.write();
        volScalarField ddt = mesh.lookupObject<volScalarField>("ddt");
        ddt.write();
        volScalarField div = mesh.lookupObject<volScalarField>("div");
        div.write();
        volScalarField laplacian = mesh.lookupObject<volScalarField>("laplacian");
        laplacian.write();
        volScalarField prod = mesh.lookupObject<volScalarField>("prod");
        prod.write();
        volScalarField sp = mesh.lookupObject<volScalarField>("sp");
        sp.write();


        

    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< nl << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
