import billing.BillingService;
import billing.InvoiceBatchJob;
import external_systems.CrmSoapClient;
import customer.CustomerAddressService;

public class LegacyBackendApplication {

    /*
     * This class represents the application entry point.
     * It wires together Billing, batch job, and external CRM communication.
     */
    public static void main(String[] args) throws Exception {

        /*
         * In a real legacy app, these values may come from application.properties.
         */
        String dbUrl = "jdbc:oracle:thin:@legacy-db:1521:ORCL";
        String dbUser = "legacy_user";
        String dbPassword = "legacy_password";
        String crmEndpoint = "http://old-crm.local/soap/customer";

        BillingService billingService = new BillingService(dbUrl, dbUser, dbPassword);

        InvoiceBatchJob invoiceBatchJob = new InvoiceBatchJob(billingService);

        CrmSoapClient crmSoapClient = new CrmSoapClient(crmEndpoint);

        CustomerAddressService customerAddressService = new CustomerAddressService(dbUrl, dbUser, dbPassword, crmSoapClient);

        /*
         * Runs nightly invoice generation/update logic.
         */
        invoiceBatchJob.runNightlyJob();

        /*
         * Sends customer address update to external CRM.
         */
        customerAddressService.updateAddress(501, "221B Baker Street");

        System.out.println("Legacy backend process completed.");
    }
}