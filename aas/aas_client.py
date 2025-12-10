import requests

#Class to represent a AAS client object and perform requests to the AAS server

class AASClient:
    def __init__(self, base_url, token=None, client_id=None, client_secret=None):
        """
        Initialise the client for the AAS api.
        :param base_url: base URL of the AAS server (without "/api") (ex: "https://aas.amlaval.fr")
        :param token: null here, will be defined in authentication call
        :param client_id: user name for authentication
        :param client_secret: password for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.client_id = client_id
        self.client_secret = client_secret

    def authenticate(self):
        """
        Authenticate with provided client_id and client_secret.
        """
        url = f"{self.base_url}/auth/oauth2/token"
        data = {
                    "grant_type": "client_credentials",
                    "scope": "*:create *:read *:update *:delete *:invoke"
                }
        response = requests.post(url, data=data, auth=(self.client_id, self.client_secret))
        response.raise_for_status()
        token_info = response.json()
        self.token = token_info.get("access_token")
        return token_info 

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    def update_submodel_element_value(self, aas_id_short, sm_id_short, se_id_short_path, value):
        """
        Endpoint to Update a submodel element.
        :param aas_id_short: idshort of the AAS
        :param sm_id_short: idshort of the submodel
        :param se_id_short_path: idshort of the submodelElement (to create or update)
        :param value: Valeur sérialisée (dict ou str)
        """
        url = f"{self.base_url}/api/shell/{aas_id_short}/aas/submodels/{sm_id_short}/submodel/submodelElements/{se_id_short_path}"
        response = requests.put(url, data=value, headers=self._headers())
        return self._handle_response(response)

    def _handle_response(self, response):
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status_code": response.status_code, "details": response.text}
