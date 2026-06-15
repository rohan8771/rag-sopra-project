package external_systems;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class CrmSoapClient {

    private final String crmEndpoint;

    /*
     * Stores the external CRM SOAP endpoint URL.
     */
    public CrmSoapClient(String crmEndpoint) {
        this.crmEndpoint = crmEndpoint;
    }

    /*
     * Sends updated customer address data to external CRM.
     */
    public void sendAddressUpdate(int customerId, String newAddress) throws Exception {

        /*
         * SOAP uses XML messages instead of JSON.
         *
         * Legacy issue:
         * XML is manually built using string concatenation.
         */
        String soapBody =
            "<soapenv:Envelope xmlns:soapenv=\"http://schemas.xmlsoap.org/soap/envelope/\">" +
                "<soapenv:Body>" +
                    "<updateAddress>" +
                        "<customerId>" + customerId + "</customerId>" +
                        "<newAddress>" + newAddress + "</newAddress>" +
                    "</updateAddress>" +
                "</soapenv:Body>" +
            "</soapenv:Envelope>";

        /*
         * Opens HTTP connection to the CRM SOAP endpoint.
         */
        URL url = new URL(crmEndpoint);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();

        /*
         * Configures this request as a SOAP/XML POST call.
         */
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "text/xml");
        connection.setDoOutput(true);

        /*
         * Sends the SOAP XML body.
         *
         * Legacy issue:
         * No retry handling, timeout config, structured logging, or error recovery.
         */
        try (OutputStream outputStream = connection.getOutputStream()) {
            outputStream.write(soapBody.getBytes());
        }

        int responseCode = connection.getResponseCode();
        System.out.println("CRM SOAP response code: " + responseCode);
    }
}