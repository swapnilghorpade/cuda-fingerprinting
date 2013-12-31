using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading;
using System.Windows.Forms;
using Obedience.Processing;
using Sonda.ACD.Packets;
using Sonda.Interop;
using Sonda.Sensors;

namespace Obedience.Host
{
    internal class ObedienceContext : ApplicationContext
    {
        private Container _components;
        private NotifyIcon _notifyIcon;

        private AcdController scanner;

        private FingerprintProcessor _processor = new FingerprintProcessor();

        internal ObedienceContext()
        {
            //Instantiate the component Module to hold everything
            _components = new System.ComponentModel.Container();
            Trace.Listeners.Add(new TextWriterTraceListener("C:\\temp\\Obedience.log"));
            
            //Instantiate the NotifyIcon attaching it to the components container and 
            //provide it an icon, note, you can imbed this resource 
            _notifyIcon = new NotifyIcon(_components);
            _notifyIcon.Icon = Resources.AppIcon;
            _notifyIcon.Text = "Obedience";
            _notifyIcon.Visible = true;

            //Instantiate the context menu and items
            var contextMenu = new ContextMenuStrip();
            var displayForm = new ToolStripMenuItem();
            var exitApplication = new ToolStripMenuItem();

            //Attach the menu to the notify icon
            _notifyIcon.ContextMenuStrip = contextMenu;

            //Setup the items and add them to the menu strip, adding handlers to be created later
            displayForm.Text = "Do something";
            displayForm.Click += mDisplayForm_Click;
            contextMenu.Items.Add(displayForm);

            exitApplication.Text = "Exit";
            exitApplication.Click += mExitApplication_Click;
            contextMenu.Items.Add(exitApplication);
            Trace.WriteLine("Obedience started");
            scanner = new AcdController(new IPEndPoint(new IPAddress(new byte[] {10, 0, 3, 220}), 5003));
            //scanner.Reboot();
            Trace.AutoFlush = true;
            scanner.ConnectedChanged += new ConnectedChangedEventHandler(scanner_ConnectedChanged);
        }

        void scanner_ConnectedChanged(bool connectedStatus)
        {
            try
            {
                if (connectedStatus)
                {
                    scanner.Sensor.Captured += Sensor_Captured;
                    var response =
                        scanner.Publisher.Publish<M162DeviceInitResponsePacket>(
                            new M162DeviceInitPacket(FamExecutiveDevice.None));
                    response =
                        scanner.Publisher.Publish<M162DeviceInitResponsePacket>(
                            new M162DeviceInitPacket(FamExecutiveDevice.Turnstile));
                }
                else
                {
                    scanner.Sensor.Captured -= Sensor_Captured;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex);
            }
        }

        void Sensor_Captured(object sender, Sonda.NET.SondaImage captureResult)
        {
            int x = 0;
            int y = 0;
            try
            {
                //Stopwatch sw = new Stopwatch();
                //sw.Start();
                // this hacky routine runs for around 1 ms - acceptable
                // completely based on ACD2 bitmap representation
                int rows = captureResult.Height;
                int columns = captureResult.Width;
                var src = captureResult.Array;
                float[] arr = new float[rows * columns];
                for (x = 0; x < columns; x++)
                {
                    for (y = 0; y < rows; y++)
                    {
                        arr[(rows-1-y)*columns + x] = 255.0f - src[y*columns + x + 1078];
                    }
                }
                //sw.Stop();
                _processor.ProcessFingerImage(arr, rows, columns);
                
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex);
            }
            Open();
        }

        private void Open()
        {
            try
            {
                scanner.Buzzer.Buzz(50);
                var response = scanner.Publisher.Publish<TurnstileResponsePacket>(new TurnstilePacket(TurnstileState.SinglePassB));

                Thread.Sleep(500);
                response = scanner.Publisher.Publish<TurnstileResponsePacket>(new TurnstilePacket(TurnstileState.Stop));
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex);
            }
        }

        private void mExitApplication_Click(object sender, EventArgs e)
        {
            _notifyIcon.Visible = false;
            scanner.Sensor.Captured -= Sensor_Captured;
            scanner.Dispose();
            ExitThreadCore();
        }

        private void mDisplayForm_Click(object sender, EventArgs e)
        {
            ExitThreadCore();
        }
    }
}
