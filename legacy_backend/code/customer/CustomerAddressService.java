package customer;

import external_systems.CrmSoapClient;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class CustomerAddressService {

    private final String dbUrl;
    private final String dbUser;
    private final String dbPassword;
    private final CrmSoapClient crmSoapClient;

    /*
     * This service updates customer address locally,
     * then sends the same update to the external CRM system.
     */
    public CustomerAddressService(
        String dbUrl,
        String dbUser,
        String dbPassword,
        CrmSoapClient crmSoapClient
    ) {
        this.dbUrl = dbUrl;
        this.dbUser = dbUser;
        this.dbPassword = dbPassword;
        this.crmSoapClient = crmSoapClient;
    }

    public void updateAddress(int customerId, String newAddress) throws Exception {

        /*
         * Legacy issue:
         * No validation of newAddress before saving it.
         */
        String sql = "UPDATE CUSTOMER SET ADDRESS = '" + newAddress + "' WHERE ID = " + customerId;

        /*
         * Legacy issue:
         * Raw JDBC Statement + string concatenation.
         */
        try (
            Connection connection = DriverManager.getConnection(dbUrl, dbUser, dbPassword);
            Statement statement = connection.createStatement()
        ) {
            statement.executeUpdate(sql);
        }

        /*
         * Legacy issue:
         * Local DB update and CRM update are not transactionally safe.
         * If CRM call fails, DB may already be updated.
         */
        crmSoapClient.sendAddressUpdate(customerId, newAddress);
    }
}