from typing import List, Any, Optional
from dataclasses import dataclass, asdict
import json
import uuid
from datetime import datetime
import base64
from aas.submodelElement import ModelType,Value, SubmodelElement

# class to create the report based on the Submodel Element AAS structure

def create_filled_report(strType:str,strTitle:str, strAnalysis:str, pdfBlob:str):
    """
        Creates the SCDT report based on the submodel element structure.
        :param strType: Type of the report (Optimization, Simulation, Stress Test)
        :param strAnalysis: Report / analysis tobe filled by the human before sending the report
        :param pdfBlob: The PDF file representation as a string
    """
    id = str(uuid.uuid4())
    report_id = f"Report_ID_{id}"
    timestamp = datetime.now().isoformat()

    collection_model = ModelType(name="SubmodelElementCollection")
    property_model = ModelType(name="Property")
    blob_model = ModelType(name="Blob")

    # Section "identification"
    id_value = Value(
        idShort="id",
        modelType=property_model,
        kind="Instance",
        value=id,
        mimeType=""
    )
    identification = Value(
        idShort="identification",
        modelType=collection_model,
        kind="Instance",
        value=[id_value],
        mimeType=""
    )

    # Section "report"
    type_value = Value(
        idShort="type",
        modelType=property_model,
        kind="Instance",
        value=strType,
        mimeType=""
    )
    timestamp_value = Value(
        idShort="timestamp",
        modelType=property_model,
        kind="Instance",
        value=timestamp,
        mimeType=""
    )
    report_processed_value = Value(
        idShort="report_processed",
        modelType=property_model,
        kind="Instance",
        value="false",
        mimeType=""
    )
    report = Value(
        idShort="report",
        modelType=collection_model,
        kind="Instance",
        value=[type_value, timestamp_value, report_processed_value],
        mimeType=""
    )

    # Section "content"
    title_value = Value(
        idShort="title",
        modelType=property_model,
        kind="Instance",
        value=strTitle,
        mimeType=""
    )
    analysis_value = Value(
        idShort="analysis",
        modelType=property_model,
        kind="Instance",
        value=strAnalysis,
        mimeType=""
    )
    attachment_value = Value(
        idShort="attachment",
        modelType=blob_model,
        kind="Instance",
        value=pdfBlob,
        mimeType="application/pdf"
    )
    content = Value(
        idShort="content",
        modelType=collection_model,
        kind="Instance",
        value=[title_value, analysis_value, attachment_value],
        mimeType=""
    )

    # Racine
    scdtReport = SubmodelElement(
        idShort=report_id,
        modelType=collection_model,
        kind="Instance",
        value=[identification, report, content]
    )

    return scdtReport

def save_to_json(scdtReport: SubmodelElement, filename: str):
    # Convert the report into a dict
    report_dict = asdict(scdtReport)
    # Write the dict into a json file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

