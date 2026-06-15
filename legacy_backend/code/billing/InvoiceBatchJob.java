package billing;

import java.sql.SQLException;

public class InvoiceBatchJob {

    private final BillingService billingService;

    /*
     * This batch job depends on BillingService to update invoices.
     */
    public InvoiceBatchJob(BillingService billingService) {
        this.billingService = billingService;
    }

    /*
     * Simulates a nightly invoice status update job.
     */
    public void runNightlyJob() throws SQLException {

        /*
         * Legacy issue:
         * Invoice IDs are manually hardcoded for this demo.
         * Real legacy systems may read these from DB queries/files.
         */
        int[] invoiceIds = {101, 102, 103};

        for (int invoiceId : invoiceIds) {
            /*
             * Legacy issue:
             * No retry handling. If one invoice update fails,
             * the whole batch may stop.
             */
            billingService.updatePaymentStatus(invoiceId, "GENERATED");
        }

        System.out.println("Nightly invoice batch completed.");
    }
}