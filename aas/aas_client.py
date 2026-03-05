import requests
import base64
import json
from pathlib import Path

class AASClient:
    def __init__(
        self, base_url, client_id=None, client_secret=None,
        token_url=None, token=None, verify_ssl=True, cert_path=None
    ):
        """
        verify_ssl: True para validar el certificado del servidor (producción)
                    False para saltarse la validación (solo local temporal)
        cert_path: ruta a certificado CA custom para test/local
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.access_token = token
        self.verify_ssl = verify_ssl
        self.cert_path = cert_path

    def _verify(self):
        """
        Decide qué usar para la verificación SSL en requests.
        """
        if self.cert_path:
            cert_file = Path(self.cert_path)
            if cert_file.exists():
                return str(cert_file.resolve())
            else:
                raise FileNotFoundError(f"Certificado no encontrado: {self.cert_path}")
        return self.verify_ssl

    # ----------------------------------------------------------------------
    # AUTHENTICATION
    # ----------------------------------------------------------------------
    def authenticate(self):
        if not self.token_url:
            raise RuntimeError("No se ha configurado token_url en AASClient")

        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "client_credentials",
            "scope": "*:create *:read *:update *:delete *:invoke"
        }
        response = requests.post(self.token_url, headers=headers, data=data, verify=self._verify())
        response.raise_for_status()
        token_info = response.json()
        self.access_token = token_info.get("access_token")
        if not self.access_token:
            raise RuntimeError("No se recibió access_token al autenticar")
        return token_info

    # ----------------------------------------------------------------------
    # INTERNAL HEADERS
    # ----------------------------------------------------------------------
    def _headers(self):
        if not self.access_token:
            raise RuntimeError("No access token. Call authenticate() first.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

    # ----------------------------------------------------------------------
    # GET / LIST / UPDATE
    # ----------------------------------------------------------------------
    def list_aas(self):
        url = f"{self.base_url}/shell/"
        response = requests.get(url, headers=self._headers(), verify=self._verify())
        return self._handle_response(response)

    def get_submodels(self, aas_id_short):
        url = f"{self.base_url}/shell/{aas_id_short}/submodels"
        response = requests.get(url, headers=self._headers(), verify=self._verify())
        return self._handle_response(response)

    def list_submodel_elements(self, aas_id_short, sm_id_short):
        url = f"{self.base_url}/shell/{aas_id_short}/aas/submodels/{sm_id_short}/submodel/submodelElements"
        response = requests.get(url, headers=self._headers(), verify=self._verify())
        return self._handle_response(response)

    def get_submodel_element(self, aas_id_short, sm_id_short, se_id_short_path):
        url = (
            f"{self.base_url}/shell/{aas_id_short}/aas/"
            f"submodels/{sm_id_short}/submodel/submodelElements/{se_id_short_path}"
        )
        response = requests.get(url, headers=self._headers(), verify=self._verify())
        return self._handle_response(response)

    def get_submodel_element_value(self, aas_id_short, sm_id_short, se_id_short_path):
        url = f"{self.base_url}/shell/{aas_id_short}/aas/submodels/{sm_id_short}/submodel/submodelElements/{se_id_short_path}/value"
        response = requests.get(url, headers=self._headers(), verify=self._verify())
        return self._handle_response(response)

    def update_submodel_element_value(self, aas_id_short, sm_id_short, se_id_short_path, value):
        body = json.dumps(value) if isinstance(value, dict) else value
        url = f"{self.base_url}/shell/{aas_id_short}/aas/submodels/{sm_id_short}/submodel/submodelElements/{se_id_short_path}/value"
        response = requests.put(url, headers=self._headers(), data=body, verify=self._verify())
        return self._handle_response(response)

    # ----------------------------------------------------------------------
    # RESPONSE HANDLER
    # ----------------------------------------------------------------------
    def _handle_response(self, response):
        try:
            response.raise_for_status()
            return response.json() if response.text else {"status": "ok"}
        except requests.exceptions.HTTPError as e:
            return {
                "error": str(e),
                "status_code": response.status_code,
                "details": response.text
            }