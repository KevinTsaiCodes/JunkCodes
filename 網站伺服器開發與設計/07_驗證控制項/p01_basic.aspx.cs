using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Validation_p01_basic : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        if (!Page.IsPostBack)
        {
            VALID_ACCOUNT.ControlToValidate = ACCOUNT.ID;
            VALID_PASSWD.ValueToCompare = ACCOUNT.Text;
            // Validation_ID.Validation_Method = Variable_ID.Variable_Type
        }
    }

    protected void SUBMIT_Click(object sender, EventArgs e)
    {

        OUTPUT.Text = "<br>Account " + ACCOUNT.Text + "<br>Password " + PASSWD.Text;
    }
}
