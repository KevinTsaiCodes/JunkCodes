<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Default.aspx.cs" Inherits="CourseSite_Default" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title></title>
    <style type="text/css">
        .auto-style1 {
            width: 100%;
        }
        .auto-style3 {
            text-align: center;
        }
        .auto-style4 {
            text-align: center;
            width: 155px;
            height: 20px;
        }
        .auto-style9 {
            text-align: center;
            width: 102px;
            height: 20px;
        }
        .auto-style13 {
            text-align: center;
            width: 102px;
        }
        .auto-style14 {
            text-align: center;
            height: 20px;
            width: 235px;
        }
        .auto-style20 {
            text-align: right;
            height: 22px;
            width: 95px;
        }
        .auto-style22 {
            text-align: center;
            height: 22px;
        }
        .auto-style23 {
            text-align: center;
            width: 107px;
            height: 22px;
        }
        .auto-style24 {
            text-align: center;
            width: 60px;
            height: 22px;
        }
        .auto-style26 {
            width: 60px;
            height: 20px;
            text-align: center;
        }
        .auto-style29 {
            text-align: center;
            width: 235px;
        }
        .auto-style30 {
            text-align: left;
        }
        .auto-style31 {
            width: 107px;
            height: 20px;
            text-align: center;
        }
        .auto-style33 {
            text-align: center;
            height: 20px;
            width: 282px;
        }
        .auto-style34 {
            font-size: x-large;
        }
        .auto-style38 {
            text-align: center;
            width: 60px;
            height: 25px;
        }
        .auto-style39 {
            text-align: center;
            width: 95px;
        }
        .auto-style40 {
            width: 107px;
            height: 18px;
            text-align: left;
        }
        .auto-style41 {
            width: 60px;
            height: 18px;
            text-align: center;
        }
        .auto-style42 {
            text-align: center;
            height: 20px;
        }
        .auto-style43 {
            font-size: xx-large;
        }
        .auto-style44 {
            width: 188px;
            height: 20px;
            text-align: center;
        }
        .auto-style45 {
            text-align: center;
            width: 188px;
        }
        .auto-style46 {
            text-align: center;
            width: 107px;
            height: 25px;
        }
    </style>
</head>
<body>

    <form id="form1" runat="server">
        <div>
            <asp:MultiView ID="MultiView1" runat="server" ActiveViewIndex="0">
                <asp:View ID="View1" runat="server">
                    <table border="1" class="auto-style1">
                        <tr>
                            <td class="auto-style44">姓名</td>
                            <td class="auto-style4" colspan="2">
                                <asp:TextBox ID="NAME" runat="server"></asp:TextBox>
                            </td>
                            <td class="auto-style9">
                                <asp:RequiredFieldValidator ID="NameValidator" runat="server" ControlToValidate="NAME" Display="Dynamic" ErrorMessage="姓名不可為空" ForeColor="Red"></asp:RequiredFieldValidator>
                            </td>
                            <td class="auto-style31">
                                <asp:TextBox ID="YEAR" runat="server" Width="60px"></asp:TextBox>
                            </td>
                            <td class="auto-style26">&nbsp;學年度</td>
                            <td class="auto-style14">
                                <asp:RadioButtonList ID="SEMESTER" runat="server" RepeatDirection="Horizontal">
                                    <asp:ListItem Value="0">上學期</asp:ListItem>
                                    <asp:ListItem Value="1">下學期</asp:ListItem>
                                </asp:RadioButtonList>
                            </td>
                            <td class="auto-style33">
                                <marquee behavior="alternate" direction="right" scrollamount="3">Today is A nice Day!</marquee></td>
                        </tr>
                        <tr>
                            <td class="auto-style45">學號</td>
                            <td class="auto-style3" colspan="2">
                                <asp:TextBox ID="ID" runat="server"></asp:TextBox>
                            </td>
                            <td class="auto-style13">
                                <asp:RequiredFieldValidator ID="IDValidator" runat="server" ControlToValidate="ID" Display="Dynamic" ErrorMessage="學號不可為空" ForeColor="Red"></asp:RequiredFieldValidator>
                            </td>
                        </tr>
                        <tr>
                            <td class="auto-style20" colspan="2">班級&nbsp; (Ex. 2A)</td>
                            <td class="auto-style22" colspan="2">
                                <asp:TextBox ID="STUCLASS" runat="server"></asp:TextBox>
                            </td>
                            <td class="auto-style22" colspan="2">
                                <asp:RegularExpressionValidator ID="STUCLASSValidator" runat="server" ControlToValidate="STUCLASS" Display="Dynamic" ErrorMessage="班級格式錯誤" ForeColor="Red" ValidationExpression="([1-4]{1})([a-z]{1}|[A-Z]{1})"></asp:RegularExpressionValidator>
                            </td>
                            <td class="auto-style22" colspan="2">
                                <marquee behavior="alternate" direction="right" scrollamount="5">It is a nice day to enroll some courses</marquee></td>
                        </tr>
                        <tr>
                            <td class="auto-style22" colspan="4">資訊工程學系必修課程</td>
                            <td class="auto-style23">資訊工程學系選修課程</td>
                            <td class="auto-style24">學分</td>
                        </tr>
                        <tr>
                            <td class="auto-style3" colspan="4" rowspan="7">
                                <div class="auto-style30">
                                    <asp:GridView ID="GridView1" runat="server" AllowPaging="True" AutoGenerateColumns="False" BackColor="White" BorderColor="#CC9966" BorderStyle="None" BorderWidth="1px" CellPadding="4" DataSourceID="SqlDataSource1" Width="285px">
                                        <Columns>
                                            <asp:TemplateField HeaderText="課程名稱" SortExpression="CourseName">
                                                <EditItemTemplate>
                                                    <asp:TextBox ID="TextBox1" runat="server" Text='<%# Bind("CourseName") %>'></asp:TextBox>
                                                </EditItemTemplate>
                                                <ItemTemplate>
                                                    <asp:Label ID="Label1" runat="server" ForeColor="Red" Text='<%# Bind("CourseName") %>'></asp:Label>
                                                </ItemTemplate>
                                            </asp:TemplateField>
                                        </Columns>
                                        <FooterStyle BackColor="#FFFFCC" ForeColor="#330099" />
                                        <HeaderStyle BackColor="#990000" Font-Bold="True" ForeColor="#FFFFCC" />
                                        <PagerStyle BackColor="#FFFFCC" ForeColor="#330099" HorizontalAlign="Center" />
                                        <RowStyle BackColor="White" ForeColor="#330099" />
                                        <SelectedRowStyle BackColor="#FFCC66" Font-Bold="True" ForeColor="#663399" />
                                        <SortedAscendingCellStyle BackColor="#FEFCEB" />
                                        <SortedAscendingHeaderStyle BackColor="#AF0101" />
                                        <SortedDescendingCellStyle BackColor="#F6F0C0" />
                                        <SortedDescendingHeaderStyle BackColor="#7E0000" />
                                    </asp:GridView>
                                </div>
                                <asp:SqlDataSource ID="SqlDataSource1" runat="server" ConnectionString="<%$ ConnectionStrings:SCHOOL_COURSEConnectionString %>" SelectCommand="SELECT * FROM [COURSE] ORDER BY [CourseName]"></asp:SqlDataSource>
                            </td>
                            <td class="auto-style23">
                                <asp:DropDownList ID="COURSE1" runat="server">
                                </asp:DropDownList>
                            </td>
                            <td class="auto-style24">
                                <asp:TextBox ID="COURSE1DEGREE" runat="server" Width="50px"></asp:TextBox>
                            </td>
                        </tr>
                        <tr>
                            <td class="auto-style23">
                                <asp:DropDownList ID="COURSE2" runat="server">
                                </asp:DropDownList>
                            </td>
                            <td class="auto-style24">
                                <asp:TextBox ID="COURSE2DEGREE" runat="server" Width="50px"></asp:TextBox>
                            </td>
                        </tr>
                        <tr>
                            <td class="auto-style31">
                                <asp:DropDownList ID="COURSE3" runat="server">
                                </asp:DropDownList>
                            </td>
                            <td class="auto-style26">
                                <asp:TextBox ID="COURSE3DEGREE" runat="server" Width="50px"></asp:TextBox>
                            </td>
                            <td class="auto-style29" rowspan="5"><span class="auto-style34">今天日期</span><asp:Calendar ID="Calendar1" runat="server" BackColor="White" BorderColor="#999999" CellPadding="4" DayNameFormat="Shortest" Font-Names="Verdana" Font-Size="8pt" ForeColor="Black" Height="210px" Width="240px">
                                <DayHeaderStyle BackColor="#CCCCCC" Font-Bold="True" Font-Size="7pt" />
                                <NextPrevStyle VerticalAlign="Bottom" />
                                <OtherMonthDayStyle ForeColor="#808080" />
                                <SelectedDayStyle BackColor="#666666" Font-Bold="True" ForeColor="White" />
                                <SelectorStyle BackColor="#CCCCCC" />
                                <TitleStyle BackColor="#999999" BorderColor="Black" Font-Bold="True" />
                                <TodayDayStyle BackColor="#CCCCCC" ForeColor="Black" />
                                <WeekendDayStyle BackColor="#FFFFCC" />
                                </asp:Calendar>
                            </td>
                        </tr>
                        <tr>
                            <td class="auto-style46">
                                <asp:DropDownList ID="COURSE4" runat="server">
                                </asp:DropDownList>
                            </td>
                            <td class="auto-style38">
                                <asp:TextBox ID="COURSE4DEGREE" runat="server" Width="50px"></asp:TextBox>
                            </td>
                        </tr>
                        <tr>
                            <td class="auto-style46">
                                <asp:DropDownList ID="COURSE5" runat="server">
                                </asp:DropDownList>
                            </td>
                            <td class="auto-style38">
                                <asp:TextBox ID="COURSE5DEGREE" runat="server" Width="50px"></asp:TextBox>
                            </td>
                        </tr>
                        <tr>
                            <td class="auto-style42" colspan="2">資訊工程學系必選修課程</td>
                        </tr>
                        <tr>
                            <td class="auto-style40">
                                <asp:CheckBoxList ID="CHOOSECOURSE" runat="server" Width="154px">
                                    <asp:ListItem Value="0">影像處理</asp:ListItem>
                                    <asp:ListItem Value="1">圖形識別</asp:ListItem>
                                    <asp:ListItem Value="2">多媒體安全</asp:ListItem>
                                    <asp:ListItem Value="3">資料壓縮</asp:ListItem>
                                </asp:CheckBoxList>
                            </td>
                            <td class="auto-style41">3</td>
                        </tr>
                        <tr>
                            <td class="auto-style39" colspan="2">
                                <asp:Button ID="ENROLL" runat="server" OnClick="ENROLL_Click" Text="點我以方便進行選課" Width="232px" />
                            </td>
                        </tr>
                    </table>
                </asp:View>
                <asp:View ID="View2" runat="server">
                    <span class="auto-style43"><strong>恭喜你/妳已選完課<br /> </strong>
                    <asp:Label ID="ENROLL_MESSAGE" runat="server" CssClass="auto-style34" ForeColor="Red"></asp:Label>
                    </span>
                </asp:View>
            </asp:MultiView>
        </div>
    </form>
</body>
</html>
