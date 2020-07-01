using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class CourseSite_Default : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        if (!Page.IsPostBack)
        {
            COURSE1.Items.Add("資訊安全導論");
            COURSE1.Items.Add("初探量子電腦");
            COURSE1.Items.Add("初探機器學習");
            COURSE2.Items.Add("資訊安全導論");
            COURSE2.Items.Add("初探量子電腦");
            COURSE2.Items.Add("初探機器學習");
            COURSE3.Items.Add("資訊安全導論");
            COURSE3.Items.Add("初探量子電腦");
            COURSE3.Items.Add("初探機器學習");
            COURSE4.Items.Add("資訊安全導論");
            COURSE4.Items.Add("初探量子電腦");
            COURSE4.Items.Add("初探機器學習");
            COURSE5.Items.Add("資訊安全導論");
            COURSE5.Items.Add("初探量子電腦");
            COURSE5.Items.Add("初探機器學習");
        }
    }
    string name, id, stuclass, choose_course="";
    int year, cd1, cd2, cd3, cd4, cd5;
    int choose_class = 0;
    protected void ENROLL_Click(object sender, EventArgs e)
    {
        name = NAME.Text;
        id = ID.Text;
        stuclass = STUCLASS.Text;

        year = int.Parse(YEAR.Text);
        cd1 = int.Parse(COURSE1DEGREE.Text);
        cd2 = int.Parse(COURSE2DEGREE.Text);
        cd3 = int.Parse(COURSE3DEGREE.Text);
        cd4 = int.Parse(COURSE4DEGREE.Text);
        cd5 = int.Parse(COURSE5DEGREE.Text);

        
        for(int i = 0; i < CHOOSECOURSE.Items.Count; i++)
        {
            if (CHOOSECOURSE.Items[i].Selected)
            {
                choose_course += CHOOSECOURSE.Items[i].Text + ",";
                choose_class += 3;
            }
        }
        MultiView1.ActiveViewIndex = 1;
        ENROLL_MESSAGE.Text += "於"+year+"學年度入學並於"+SEMESTER.SelectedItem.Text+"學期的<br>"+"名字: " + name + "，且學號為 " + id + "，班級為 " + stuclass + "歡迎妳<br>";
        ENROLL_MESSAGE.Text += "總共修了系選" + (cd1 + cd2 + cd3 + cd4 + cd5) + "學分，以及必選修" + choose_class + "學分";
    }

}
