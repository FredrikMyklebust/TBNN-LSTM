/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2020 OpenCFD Ltd.
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

#include "frozensolver.H"
#include "fvOptions.H"
#include "bound.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void frozensolver<BasicTurbulenceModel>::correctNut()
{
    this->nut_ = k_DNS_/omega_;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
frozensolver<BasicTurbulenceModel>::frozensolver
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    Cmu_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    beta_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta",
            this->coeffDict_,
            0.072
        )
    ),
    gamma_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma",
            this->coeffDict_,
            0.52
        )
    ),
    alphaK_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega",
            this->coeffDict_,
            0.5
        )
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("krans", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    omega_old_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            Foam::name((this->runTime_.time()-this->runTime_.deltaT()).value()),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    omega_old_old_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            Foam::name((this->runTime_.time()-2*this->runTime_.deltaT()).value()),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    k_DNS_
    (
        IOobject
        (
            IOobject::groupName("kdns", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    k_DNS_old_
    (
        IOobject
        (
            IOobject::groupName("kdns", alphaRhoPhi.group()),
            Foam::name((this->runTime_.time()-this->runTime_.deltaT()).value()),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    k_DNS_old_old_
    (
        IOobject
        (
            IOobject::groupName("kdns", alphaRhoPhi.group()),
            Foam::name((this->runTime_.time()-2*this->runTime_.deltaT()).value()),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),
    tauij_DNS_
    (
        IOobject
        (
            "Rdns",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    ),

    aij_DNS_
    (
        IOobject
        (
            "aij_DNS",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        tauij_DNS_ - (twoThirdsI)*k_DNS_
    ),
    aijBoussinesq_
    (
        IOobject
        (
            "aijBoussinesq",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        0.0*symm(fvc::grad(this->U_))*this->nut_
    ),
    aijDelta_
    (
        IOobject
        (
            "aijDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        0.0*symm(fvc::grad(this->U_))*this->nut_
    ),
    bijDelta_
    (
        IOobject
        (
            "bijDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        0.0*symm(fvc::grad(this->U_))/omega_
    ),
    kDeficit_
    (
        IOobject
        (
            "kDeficit",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("kDeficit", dimensionSet(0,2,-3,0,0,0,0), 0.0)
    ),
    ddt_
    (
        IOobject
        (
            "ddt",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("ddt", dimensionSet(0,2,-3,0,0,0,0), 0.0)
    ),
    div_
    (
        IOobject
        (
            "div",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("div", dimensionSet(0,2,-3,0,0,0,0), 0.0)
    ),
    laplacian_ (
        IOobject
        (
            "laplacian",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("laplacian", dimensionSet(0,2,-3,0,0,0,0), 0.0)
    ),
    prod_
    (
        IOobject
        (
            "prod",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("prod", dimensionSet(0,2,-3,0,0,0,0), 0.0)
    ),
    sp_
    (
        IOobject
        (
            "sp",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("sp", dimensionSet(0,2,-3,0,0,0,0), 0.0)
    ),

    bij_DNS_
    (
        IOobject
        (
            "bij_DNS",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        aij_DNS_ / 2.0 / (k_DNS_)
    ),
    PkDNS_
    (
        IOobject
        (
            "PkDNS",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("PkDNS", dimensionSet(0,2,-3,0,0,0,0), 0.0)
    )
    {
    // Add the dictionary reading code here
    IOdictionary dict
    (
        IOobject
        (
            "frozenDict",
            this->runTime_.system(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
        // Extract parameters from the dictionary
    scalar usek = dict.subDict("parameters").lookupOrDefault<scalar>("usek", 1.0);
    //scalar param2 = dict.subDict("parameters").lookupOrDefault<scalar>("param2", 0.0);
    //word param3 = dict.subDict("parameters").lookupOrDefault<word>("param3", "defaultString");

    Info << "usek: " << usek << nl;



    bound(k_DNS_, this->kMin_);
    bound(omega_, this->omegaMin_);

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool frozensolver<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        beta_.readIfPresent(this->coeffDict());
        gamma_.readIfPresent(this->coeffDict());
        alphaK_.readIfPresent(this->coeffDict());
        alphaOmega_.readIfPresent(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void frozensolver<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }
    // Read the K_DNS from the previous time folder. This appraoach is not optimal but should work for now
    // Now lets set the k_DNS_.oldTime() to the k_DNS_old_ so we can get a time derivative

    k_DNS_.oldTime() = k_DNS_old_;
    k_DNS_.oldTime().oldTime() = k_DNS_old_old_;
    omega_.oldTime() = omega_old_;
    omega_.oldTime().oldTime() = omega_old_old_;

    // read the trubulence values
    // Read k_DNS from the current time directory
/*
    tauij_DNS_ = volSymmTensorField
(
    IOobject
    (
        "Rdns",
        this->runTime_.timeName(),
        this->mesh_,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    this->mesh_
);

    // Defining kDeficit_ as volScalarField
    k_DNS_ = volScalarField
    (
        IOobject
        (
            "kdns",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    );

    // Reading U_DNS as volVectorField
    const volVectorField U_DNS
    (
        IOobject
        (
            "Udns",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    );
    */

// Old:     aij_DNS_ = tauij_DNS_ - (twoThirdsI)*k_DNS_;
//lets see if tau = u'u' and not -u'u'

    aij_DNS_ = tauij_DNS_ - (twoThirdsI)*k_DNS_;
    bij_DNS_ = aij_DNS_ / 2.0 / (k_DNS_);

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    const volScalarField& nut = this->nut_;
    
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))().v()
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volScalarField S2(2*magSqr(symm(tgradU())));
    //info this works
    volScalarField G2(this->GName(), nut*S2 -(aijDelta_ && tgradU()));
    PkDNS_ = G2;
    /*
    const volScalarField::Internal GbyNu  
    (
        tgradU().v() && dev(twoSymm(tgradU().v()))
    );
    const volScalarField::Internal G(this->GName(), nut()*GbyNu);
  

    // Production term from HiFi dataset
    PkDNS_ = -tauij_DNS_ && tgradU();
    */
    tgradU.clear();

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();

     dimensionedScalar nutSmall
    (
        "nutSmall",
        dimensionSet(0, 2, -1, 0, 0, 0 ,0),
        1e-10
    );
    //Info << "Starting Omega solve" << endl;
    //Info << "nut" << nut << endl;
    //Info << "G2" << G2 << endl;
    //Info << "k/omega" << k_DNS_/omega_ << endl;
    //Info << "nut" << nut << endl;
    // Turbulence specific dissipation rate equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhi, omega_)
      - fvm::laplacian(alpha*rho*DomegaEff(), omega_)
     ==
        gamma_*alpha()*rho()*G2/(nut +nutSmall)//omega_/k_DNS_
      + gamma_*alpha()*rho()*kDeficit_/(nut +nutSmall)//*usek
      - fvm::SuSp(((2.0/3.0)*gamma_)*alpha()*rho()*divU, omega_)
      - fvm::Sp(beta_*alpha()*rho()*omega_(), omega_)
      + fvOptions(alpha, rho, omega_)

    );
    // // Conditionally add the usek term if it's not zero
    // if (usek != 0)
    // {
    //     omegaEqn.ref() += gamma_*alpha()*rho()*kDeficit_/(nut +nutSmall);
    // }

    //Info << "Relaxing Omega" << endl;
    omegaEqn.ref().relax();
    fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    fvOptions.correct(omega_);
    bound(omega_, this->omegaMin_);

    bound(k_DNS_, this->kMin_);
    // kUDeficit_ refers to R term (correction term in kU-equation)
    // Turbulent kinetic energy equation
    //Info << "Calculating KDefecit_" << endl;
    ddt_ = fvc::ddt(rho*k_DNS_);
    div_ = fvc::div(alphaRhoPhi, k_DNS_);
    laplacian_ = fvc::laplacian(alpha()*rho()*DkEff(), k_DNS_);
    prod_ = alpha()*rho()*G2;
    sp_ = fvc::Sp(Cmu_*alpha()*rho*omega_, k_DNS_);


    kDeficit_ =   fvc::ddt(rho*k_DNS_)
                + fvc::div(alphaRhoPhi, k_DNS_)
                - fvc::laplacian(alpha()*rho()*DkEff(), k_DNS_)
                - alpha()*rho()*G2
                //+ (2.0/3.0)*alpha*rho*divU*k_LES_ // Incompressible: divU = 0
                + fvc::Sp(Cmu_*alpha()*rho*omega_, k_DNS_); // betastar = C_mu

    this->omega_ = omega_;
    correctNut();
    
    // Calculate bijUDelta, the model correction term for RST equation
    aijBoussinesq_ = -nut*twoSymm(fvc::grad(U));
    aijDelta_ = aij_DNS_ - aijBoussinesq_;
    bijDelta_ = aijDelta_ / 2.0 / k_DNS_;
    bijDelta_.correctBoundaryConditions();
    //bijDelta_ = bij_DNS_ + nut / k_DNS_ * symm(fvc::grad(this->U_));
    //print out bijDelta_ for debugging
    //Info << "kDeficit_ = " << kDeficit_.internalField()[1] << endl;


    //k_DNS_.oldTime() = k_DNS_;
    //Info << "k oldtime = " << k_DNS_.oldTime() << endl;
    //kDeficit_.write();
    //bijDelta_.write();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //

