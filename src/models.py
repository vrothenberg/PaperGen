# src/models.py

from pydantic import BaseModel, Field
from typing import List, Union


# Search queries response format
class SearchQuery(BaseModel):
    section: str = Field(..., description="The section of the outline the query corresponds to.")
    query: str = Field(..., description="The search query for this section.")


# Knowledgebase article format
class ReferenceItem(BaseModel):
    reference_number: int = Field(..., description="Sequential reference number.")
    authors: str = Field(..., description="Author(s) of the reference.")
    year: str = Field(..., description="Publication year.")
    title: str = Field(..., description="Title of the reference.")
    journal_source: str = Field(..., description="Journal or source of publication.")
    url_doi: str = Field(..., description="URL or DOI of the reference.")


class FAQItem(BaseModel):
    question: str = Field(..., description="A frequently asked question about the condition.")
    answer: str = Field(..., description="A concise and informative answer to the question.")


class OverviewSection(BaseModel):
    heading: str = Field(default="Overview", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="A high-level introduction to the topic, explaining its significance and impact, key statistics, and facts.")


class KeyFactsSection(BaseModel):
    heading: str = Field(default="Key Facts", description="Heading of the section.")  # Corrected
    content: List[str] = Field(..., description="Notable statistics or key data, presented as a list of strings.")


class SymptomsSection(BaseModel):
    heading: str = Field(default="Symptoms", description="Heading of the section.")  # Corrected
    content: List[str] = Field(..., description="Common signs and symptoms, presented as a list of strings.")


class TypesSection(BaseModel):
    heading: str = Field(default="Types", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Classifications or subtypes, each with a brief explanation. Types and subtypes may use nested subheadings (###, ####).")


class CausesSection(BaseModel):
    heading: str = Field(default="Causes", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Explanation of underlying mechanisms or causes, including primary causes.")


class RiskFactorsSection(BaseModel):
    heading: str = Field(default="Risk Factors", description="Heading of the section.")  # Corrected
    content: List[str] = Field(..., description="Factors that increase susceptibility, including lifestyle, genetic, or environmental risk factors.")


class DiagnosisSection(BaseModel):
    heading: str = Field(default="Diagnosis", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Outline of the diagnostic process, including medical history, clinical symptoms, tests, or imaging tools. Can include subheadings (###) for specific methods.")


class PreventionSection(BaseModel):
    heading: str = Field(default="Prevention", description="Heading of the section.")  # Corrected
    content: List[str] = Field(..., description="Practical advice for risk reduction, with evidence-based recommendations.")


class SpecialistToVisitSection(BaseModel):
    heading: str = Field(default="Specialist to Visit", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Healthcare providers involved in diagnosis and treatment, and their roles.")


class TreatmentSection(BaseModel):
    heading: str = Field(default="Treatment", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Description of medical and therapeutic interventions.")


class HomeCareSection(BaseModel):
    heading: str = Field(default="Home-Care", description="Heading of the section.")  # Corrected
    content: List[str] = Field(..., description="Tips for self-management, such as lifestyle adjustments, routines, or home remedies.")


class LivingWithSection(BaseModel):
    heading: str = Field(default="Living With", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Guidance for long-term management, including emotional, social, or physical adaptation strategies.")


class ComplicationsSection(BaseModel):
    heading: str = Field(default="Complications", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Discussion of potential health challenges if the condition is untreated or poorly managed.")


class AlternativeTherapiesSection(BaseModel):
    heading: str = Field(default="Alternative Therapies", description="Heading of the section.")  # Corrected
    content: str = Field(..., description="Summary of non-conventional approaches.")


class FAQsSection(BaseModel):
    heading: str = Field(default="FAQs", description="Heading of the section.")  # Corrected
    content: List[FAQItem] = Field(..., description="A list of frequently asked questions and their corresponding answers.")


class ReferencesSection(BaseModel):
    heading: str = Field(default="References", description="Heading of the section.")  # Corrected
    content: List[ReferenceItem] = Field(..., description="A list of references.")


class Outline(BaseModel):
    title: str = Field(..., description="The main heading of the article.")
    subtitle: str = Field(..., description="A concise introductory phrase summarizing the condition.")
    sections: List[Union[
        OverviewSection, KeyFactsSection, SymptomsSection, TypesSection,
        CausesSection, RiskFactorsSection, DiagnosisSection, PreventionSection,
        SpecialistToVisitSection, TreatmentSection, HomeCareSection,
        LivingWithSection, ComplicationsSection, AlternativeTherapiesSection,
        FAQsSection, ReferencesSection
    ]] = Field(..., description="A list of sections in the article.")
