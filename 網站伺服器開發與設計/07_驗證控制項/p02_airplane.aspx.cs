using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class ch07_p02_airplane : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        if (!Page.IsPostBack)
        {
            PRICE.Text = "";
        }
    }

    protected void SUBMIT_Click(object sender, EventArgs e)
    {
        if (Page.IsValid)
        {
            int price = int.Parse(SUMOFTICKETS.Text);
            price *= 100;
            PRICE.Text += "您花費是: " + price.ToString() + "元\n";
        }
    }

    protected void CLEAR_Click(object sender, EventArgs e)
    {
        PRICE.Text = "";
    }
}
