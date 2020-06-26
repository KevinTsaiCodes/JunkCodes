<%@ Page Language="C#" AutoEventWireup="true" CodeFile="p01_Airplane.aspx.cs" Inherits="ch08_p01_Airplane" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
    <style type="text/css">
        .auto-style1 {
            text-align: center;
        }
        .auto-style2 {
            font-size: x-large;
        }
        .auto-style3 {
            font-size: x-large;
            color: #FF3300;
            background-color: #C0C0C0;
        }
        .auto-style4 {
            font-size: xx-large;
        }
    </style>
</head>
<body>
    <form id="form1" runat="server">
        <div class="auto-style1">
            <asp:GridView ID="GridView1" runat="server" AutoGenerateColumns="False" DataSourceID="SqlDataSource1" Width="156px">
                <Columns>
                    <asp:TemplateField HeaderText="起點" SortExpression="STARTING_POINT">
                        <EditItemTemplate>
                            <asp:TextBox ID="TextBox1" runat="server" Text='<%# Bind("STARTING_POINT") %>'></asp:TextBox>
                        </EditItemTemplate>
                        <ItemTemplate>
                            <asp:Label ID="Label1" runat="server" Text='<%# Bind("STARTING_POINT") %>'></asp:Label>
                        </ItemTemplate>
                    </asp:TemplateField>
                    <asp:TemplateField HeaderText="終點" SortExpression="DESTINATION">
                        <EditItemTemplate>
                            <asp:TextBox ID="TextBox2" runat="server" Text='<%# Bind("DESTINATION") %>'></asp:TextBox>
                        </EditItemTemplate>
                        <ItemTemplate>
                            <asp:Label ID="Label2" runat="server" Text='<%# Bind("DESTINATION") %>'></asp:Label>
                        </ItemTemplate>
                    </asp:TemplateField>
                </Columns>
            </asp:GridView>
        </div>
        <asp:SqlDataSource ID="SqlDataSource1" runat="server" ConnectionString="<%$ ConnectionStrings:TRANSPORATIONConnectionString1 %>" SelectCommand="SELECT * FROM [airplane]"></asp:SqlDataSource>
        <br />
        <asp:Label runat="server" CssClass="auto-style2" Text="以上每張票均一價100元，請問要買幾張"></asp:Label>
&nbsp;<asp:TextBox ID="SUMOFTICKETS" runat="server" Width="30px"></asp:TextBox>
        <br />
        <br />
        <asp:Button ID="SUBMIT" runat="server" CssClass="auto-style3" OnClick="SUBMIT_Click" Text="點我購買" Width="120px" />
&nbsp;
        <asp:Button ID="CLEAR" runat="server" CssClass="auto-style3" OnClick="CLEAR_Click" Text="點我取消或是重新購買" Width="260px" />
        <br />
        <asp:Label ID="PRICE" runat="server" CssClass="auto-style4"></asp:Label>
    </form>
</body>
</html>
