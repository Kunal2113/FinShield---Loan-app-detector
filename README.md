🛡️ FinShield

Digital Lending Risk Intelligence & Safety Hub

FinShield is a web-based digital lending safety platform designed tohelp users evaluate the credibility and potential risk of instant-loanapplications before installing or using them.

Predatory loan applications can expose borrowers to excessivepermissions, unclear disclosures, misleading information, and abusiverecovery practices. FinShield brings multiple safety signals togetherinto a simple, explainable interface so users can make more informeddecisions.

Live Demo:https://finshield---loan-app-detectorgit-bywwrgim8e5mbnpnzn3qme.streamlit.app/

📌 Problem Statement

RBI-flagged predatory loan apps can harass and coerce borrowers, whileusers often have no simple way to assess an app's risk before installingit.

Existing approaches have several limitations:

Manual verification is time-consuming.

Users may struggle to interpret app permissions and lendingdisclosures.

App ratings alone do not provide enough context.

Reviews and complaints are rarely analyzed together with other riskindicators.

Users need a simple explanation of why an app may be risky.

FinShield addresses this gap through an intelligent, multi-factor riskassessment and a user-friendly safety hub.

💡 Our Solution

FinShield evaluates digital lending applications using multipleindicators and presents the results through an intuitive risk assessmentinterface.

The platform can provide:

Risk score and risk level

Legitimate / predatory-oriented assessment

RBI/regulatory status indicators

Terms and disclosure assessment

Review sentiment indicators

Harassment/recovery-related review mentions

Key risk drivers and explanations

Verified website information where available

Borrower safety profiling

Lending product rankings

Financial advisory calculations

RBI digital lending safety guidance

The goal is not simply to display a score, but to make the underlyingrisk signals easier for a user to understand.

✨ Key Features

1. 🔍 App Risk Scorer

The primary FinShield module allows users to evaluate a lendingapplication.

Users can:

Select a pre-analyzed lending application.

Audit an unlisted application.

Provide a Play Store link, website URL, APK/package identifier, orsupported app reference.

View an overall risk verdict.

See a visual Riskometer.

Review individual risk indicators and explanations.

The current interface also supports auditing web domains/unlistedsources and displays an active assessment status while the analysis isbeing performed.

2. 📊 Riskometer & Explainable Verdict

Instead of presenting only raw data, FinShield converts the assessmentinto an easy-to-understand risk level.

Example verdict categories include:

Low

Low--Moderate

Moderate

Moderately High

High

Very High

The dashboard can also explain important risk drivers, such as:

Regulatory status

Terms/disclosure quality

Harassment-related review mentions

Strongly negative reviews

Review sentiment

Review characteristics

Other compliance or permission concerns

3. 👤 Borrower Safety Profiler

The Borrower Safety Profiler evaluates a user's lending andprivacy-safety habits through a questionnaire.

It considers factors such as:

Frequency of instant-loan usage

Whether users grant contacts/gallery permissions

Whether lenders are verified through RBI resources

Availability of an emergency fund

The result is a Borrower Safety Profile with a Safety Index and aneasy-to-understand borrower category.

For example:

Prudent Borrower

Vulnerable Borrower

This module is intended to encourage safer borrowing and better privacypractices.

4. 📈 Digital Lending App Rankings

The Product Rankings module provides an evaluated database of lendingapplications.

Users can search by:

App name

Package ID / App ID

The interface presents evaluated applications in a searchable table,helping users compare lending apps before making a decision.

5. 🧮 Financial Advisory Calculators

FinShield includes financial tools designed to help users understand theactual cost of borrowing.

Personal Loan Prepayment Calculator

Calculates:

Monthly EMI

Total interest payable

Potential prepayment interest savings

Hidden Fees & True APR Detector

Helps identify the effective cost of short-term borrowing byconsidering:

Disbursed amount

Repayment amount

Loan duration

Extra fees and interest

Annualized APR

The tool can flag unusually high annualized costs and display a warningwhen the calculated APR crosses a defined safety threshold.

6. 📜 RBI Digital Lending Guidelines

The RBI Guidelines section provides a simplified safety checklistcovering important digital lending practices, including:

Prohibited access to sensitive personal data

Key Fact Statement (KFS)

Direct bank-account transfer requirements

Grievance redressal information

Verification through RBI resources

Cybercrime reporting guidance

The section is designed to help users understand important safety checkswithout needing to interpret lengthy regulatory documents themselves.

7. 🌙 Dark Mode & User-Friendly UI

FinShield uses a modern dashboard-style interface with:

Dark mode

Modular navigation

Visual risk indicators

Responsive information cards

Simple explanations

Clear warning states

The interface is designed for users with minimal technical knowledge.

⚙️ How FinShield Works

                    ┌──────────────────────┐
                    │   User Input         │
                    │ App / URL / Package  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Acquisition     │
                    │ Play Store / Web     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │ & Feature Engineering│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Risk Analysis Engine │
                    │ ML + Review Analysis │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Risk Score & Level   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Explainable Results  │
                    │ + Safety Guidance    │
                    └──────────────────────┘

🤖 AI / ML Component

The original project architecture uses an ML-based classificationpipeline to distinguish between legitimate and potentially predatorylending applications.

The analysis combines multiple app-level signals rather than relyingonly on a single rating.

The project presentation reports:

80+ manually verified loan applications

88% model accuracy

Approximately 2 seconds prediction time per app

These figures describe the project evaluation presented duringdevelopment and may change as the dataset and model are expanded.

🧠 Review & Sentiment Analysis

User reviews provide important signals about the real-world experienceof borrowers.

FinShield uses sentiment analysis to derive indicators such as:

Average review tone

Strongly negative review percentage

Harassment-related mentions

Review length characteristics

These signals are combined with other app indicators to provide abroader risk assessment.

🛠️ Technology Stack

Technology                          Purpose

Python                          Core application and ML pipeline

Streamlit                       Interactive web application

Scikit-learn                    Model training, pipeline,evaluation and prediction

VADER Sentiment                 Review sentiment analysis

Pandas                          Data processing and analysis

NumPy                           Numerical computation

Google Play metadata scraping   App metadata, permissions, installsand review collection

GitHub                          Version control and collaboration

The project presentation also describes a custom data collectionpipeline for Google Play Store metadata, permissions, install counts anduser reviews.

🔄 Development Pipeline

Google Play Store / Public Sources
              ↓
       Data Collection
              ↓
          Dataset
              ↓
      Data Preprocessing
              ↓
     Feature Engineering
              ↓
       Pattern Analysis
              ↓
        ML Pipeline
              ↓
      Model Evaluation
              ↓
        Deployment
              ↓
        FinShield UI

📊 Current FinShield Modules

Module                        Purpose

🔍 App Risk Scorer            Evaluate lending-app risk👤 Borrower Safety Profiler   Assess borrower safety habits📊 Product Rankings           Search and compare evaluated apps🧮 Advisory Calculators       Understand loan costs and APR📜 RBI Guidelines             Provide lending safety guidance

🚀 Getting Started

Prerequisites

Make sure Python is installed on your system.

Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>

Create a virtual environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Run FinShield

streamlit run app.py

The exact entry-point filename may differ depending on the repositorystructure.

🌐 Live Demo

FinShield Web App

https://finshield---loan-app-detectorgit-bywwrgim8e5mbnpnzn3qme.streamlit.app/

🎯 Objectives

FinShield aims to:

Help users identify potentially risky lending applications beforeinstallation.

Make complex lending-safety indicators easier to understand.

Encourage users to verify lenders through official regulatoryresources.

Highlight privacy and harassment-related risks.

Help borrowers understand the real cost of short-term loans.

Promote transparency and safer participation in the digital lendingecosystem.

🌍 Impact

For End Users

Identify risky applications before installation.

Make more informed borrowing decisions.

Reduce exposure to fraud, privacy violations and harassment.

Understand lending costs more clearly.

For the Financial Ecosystem

Encourage responsible digital lending.

Promote transparency and trust.

Provide scalable AI-assisted decision support.

Support continuous monitoring and model improvement.

For Regulatory Support

FinShield can complement existing verification initiatives by helpingusers identify suspicious indicators and directing them toward officialverification and reporting resources.

🔮 Future Scope

The project roadmap includes:

Continuous dataset expansion

Automated end-to-end ML pipeline

Easy model replacement and retraining

Integration with live verification sources

Browser extension

Android application

Integration with app stores and fintech platforms

Larger-scale real-time monitoring

📚 Data & References

FinShield's project material references:

Google Play Store --- app metadata, permissions, install counts,ratings and reviews.

Reserve Bank of India (RBI) --- digital lending guidelines andregulatory information.

Public reports and regulatory references concerningpredatory/banned loan applications.

Custom dataset created using Google Play Store data collection,manual verification and domain-specific feature engineering.

⚠️ Disclaimer

FinShield is an informational and decision-support tool, not afinancial or legal authority.

A risk score should not be treated as definitive proof that anapplication is legitimate, fraudulent, or illegal. Users shouldindependently verify lenders through official regulatory sources beforesharing sensitive information or entering into a financial agreement.

👥 Team

Team Name: Synapesex

College: Hindustan College of Science and Technology, Farah, Mathura

Team Leader: Oorvi Kulshreshtha

⭐ Vision

Make digital lending safer, more transparent, and easier tounderstand --- before a borrower clicks "Install."

FinShield combines risk analysis, borrower awareness, financialcalculations and regulatory guidance into one accessible digital lendingsafety hub.
